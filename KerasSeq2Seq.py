from keras.models import Model
from keras.layers import Input, Dense, CuDNNGRU, Dropout
import numpy as np


class DeepSeq2SeqModel:
    def __init__(self, in_shape, out_shape, enc_lyrs=[16, 16], dec_lyrs=[16, 16],
                 optimizer='RMSprop', loss='mse', use_dropout=True, dropout=0.1):

        self.in_shape = in_shape
        self.out_shape = out_shape

        encoder_input = Input(shape=(None, self.in_shape[1]))

        encoder_layers = []

        for i in enc_lyrs[:-1]:
            if use_dropout:
                encoder_layers.append(Dropout(dropout))
            encoder_layers.append(CuDNNGRU(i, return_sequences=True))

        if use_dropout:
            encoder_layers.append(Dropout(dropout))
        encoder_layers.append(CuDNNGRU(enc_lyrs[-1], return_state=True))

        x = encoder_layers[0](encoder_input)

        for i in range(1, len(encoder_layers)-1):
            x = encoder_layers[i](x)

        # Grab the state of the final layer
        _, state_h = encoder_layers[-1](x)

        # Feedback input for decoder
        self.decoder_input = Input(shape=(None, self.out_shape[1]))

        self.decoder_layers = [CuDNNGRU(dec_lyrs[0], return_sequences=True)]

        for i in range(1, len(dec_lyrs)):
            self.decoder_layers.append(CuDNNGRU(dec_lyrs[i], return_sequences=True))

        x = self.decoder_layers[0](self.decoder_input, initial_state=state_h)

        for i in range(1, len(self.decoder_layers)):
            x = self.decoder_layers[i](x)

        self.decoder_dense = Dense(self.out_shape[1], activation='linear')
        decoder_output = self.decoder_dense(x)

        # Full end-to-end model
        self.model = Model([encoder_input, self.decoder_input], decoder_output)
        self.model.compile(optimizer=optimizer, loss=loss)

        # Just encoder model Input to output state
        self.encoder_model = Model(encoder_input, state_h)

    def train(self, x_train, y_train, batch_size=10, epochs=50, verbose=2, start_tok=0.0):
        # Need to add start to token to all the samples in the training set as training feedback into the decoder
        y_fb = [np.concatenate([np.full((1, y_train.shape[2]), start_tok), i[:-1, :]], axis=0) for i in y_train]
        y_fb = np.rollaxis(np.dstack(y_fb), -1)

        # Fit the full encoder-decoder model in conjunction
        return self.model.fit([x_train, y_fb], y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_split=0.2,
                              verbose=verbose,
                              shuffle=True)

    def predict(self, samples):

        print("Building prediction decoder network")
        # Feedback outputs back into the decoder
        decoder_input = Input(batch_shape=(1, 1, self.out_shape[1]))
        decoder_state_input = Input(shape=(self.decoder_layers[0].output_shape[-1],))

        pred_lyrs = [CuDNNGRU(self.decoder_layers[0].output_shape[-1],
                              return_sequences=True,
                              return_state=True,
                              weights=self.decoder_layers[0].get_weights())]

        x, state_d = pred_lyrs[0](decoder_input, initial_state=decoder_state_input)

        for i in range(1, len(self.decoder_layers)):
            pred_lyrs.append(CuDNNGRU(self.decoder_layers[i].output_shape[-1],
                                      return_sequences=True,
                                      stateful=True,
                                      weights=self.decoder_layers[i].get_weights()))
            x = pred_lyrs[i](x)

        decoder_output = self.decoder_dense(x)

        decoder_model = Model([decoder_input, decoder_state_input], [decoder_output, state_d])

        print("Running predictions")

        res = []

        for i in samples:
            state_value = self.encoder_model.predict(i[np.newaxis, ...])
            target_seq = np.zeros(shape=(1, 1, self.out_shape[1]))
            inner_res = []

            for _ in range(self.out_shape[0]):
                output_tokens, state_value = decoder_model.predict([target_seq, state_value])
                inner_res.append(output_tokens[0, 0])
                target_seq = output_tokens

            res.append(inner_res)
            for j in pred_lyrs[1:]:
                j.reset_states()

        return np.concatenate([res])
