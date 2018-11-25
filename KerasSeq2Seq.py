from keras.models import Model
from keras.layers import Input, Dense, GRU, Dropout, CuDNNGRU
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K


class DeepSeq2SeqModelB:
    def __init__(self, hidden_dim, in_shape, out_shape, optimizer='RMSprop', loss='mse', dropout=0.1):

        self.hidden_dim = hidden_dim
        self.in_shape = in_shape
        self.out_shape = out_shape

        # Desired encoder layer structure
        encoder_layers = [Input(shape=(None, self.in_shape[1])),
                          CuDNNGRU(hidden_dim, return_sequences=True),#, activity_regularizer=regularizers.l1(10e-5)),
                          # Dropout(dropout),
                          CuDNNGRU(int(hidden_dim/2), return_sequences=True),#activity_regularizer=regularizers.l1(10e-4)),
                          # Dropout(dropout),
                          CuDNNGRU(int(hidden_dim / 4), return_state=True)]#, activity_regularizer=regularizers.l1(10e-4))]

        # Wire up the layer list
        for i in range(1, len(encoder_layers)):
            encoder_layers[i] = encoder_layers[i](encoder_layers[i-1])

        # Grab the state of the final layer
        _, state_h = encoder_layers[-1]

        # Feedback outputs back into the decoder
        self.decoder_input = Input(shape=(None, self.out_shape[1]))

        # Desired encoder layer structure
        self.decoder_gru_1 = CuDNNGRU(int(hidden_dim/4), return_sequences=True)
        decoder_gru_1a = self.decoder_gru_1(self.decoder_input, initial_state=state_h)

        self.decoder_gru_2 = CuDNNGRU(int(hidden_dim/2), return_sequences=True)
        decoder_gru_2a = self.decoder_gru_2(decoder_gru_1a)

        self.decoder_gru_3 = CuDNNGRU(int(hidden_dim), return_sequences=True)
        decoder_output = self. decoder_gru_3(decoder_gru_2a)

        self.decoder_dense = Dense(self.out_shape[1], activation='linear')
        decoder_output = self.decoder_dense(decoder_output)

        # Full end-to-end model
        self.model = Model([encoder_layers[0], self.decoder_input], decoder_output)
        self.model.compile(optimizer=optimizer, loss=loss)

        # Just encoder model Input to output state
        self.encoder_model = Model(encoder_layers[0], state_h)

    def train(self, x, y, batch_size=10, epochs=50, verbose=2):
        yi = [np.concatenate([np.zeros((1, y.shape[2])), i[:-1, :]], axis=0) for i in y]
        yi = np.rollaxis(np.dstack(yi), -1)
        history = self.model.fit([x, yi], y,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=0.2,
                                 verbose=verbose,
                                 shuffle=True)

        plt.figure(figsize=(20, 5))
        plt.semilogy(history.history['loss'])
        plt.semilogy(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def predict(self, x):

        print("Building decoder network")
        # Feedback outputs back into the decoder
        decoder_input = Input(batch_shape=(1, 1, self.out_shape[1]))
        decoder_state_input = Input(shape=(int(self.hidden_dim / 4),))

        decoder_gru_1 = CuDNNGRU(int(self.hidden_dim/4),
                                 return_sequences=True,
                                 return_state=True,
                                 weights=self.decoder_gru_1.get_weights())
        decoder_gru_1a, state_d = decoder_gru_1(decoder_input, initial_state=decoder_state_input)

        decoder_gru_2 = CuDNNGRU(int(self.hidden_dim / 2),
                                 return_sequences=True,
                                 stateful=True,
                                 weights=self.decoder_gru_2.get_weights())
        decoder_gru_2a = decoder_gru_2(decoder_gru_1a)

        decoder_gru_3 = CuDNNGRU(int(self.hidden_dim),
                                 return_sequences=True,
                                 stateful=True,
                                 weights=self.decoder_gru_3.get_weights())
        decoder_output = decoder_gru_3(decoder_gru_2a)

        decoder_output = self.decoder_dense(decoder_output)

        decoder_model = Model([decoder_input, decoder_state_input], [decoder_output, state_d])

        print("Running predictions")

        res = []

        for i in x:
            state_value = self.encoder_model.predict(i[np.newaxis, ...])
            target_seq = np.zeros(shape=(1, 1, self.out_shape[1]))
            inner_res = []

            for _ in range(self.out_shape[0]):
                output_tokens, s0 = decoder_model.predict([target_seq, state_value])
                inner_res.append(output_tokens[0, 0])
                target_seq = output_tokens
                state_value = s0

            res.append(inner_res)
            decoder_gru_2.reset_states()
            decoder_gru_3.reset_states()

        return np.concatenate([res])


class DeepSeq2SeqModel:
    def __init__(self, hidden_dim, in_shape, out_shape, optimizer='RMSprop', loss='mse', dropout=0.1):

        self.hidden_dim = hidden_dim
        self.in_shape = in_shape
        self.out_shape = out_shape

        # Desired encoder layer structure
        encoder_layers = [Input(shape=(None, self.in_shape[1])),
                           GRU(hidden_dim, return_sequences=True),#, activity_regularizer=regularizers.l1(10e-5)),
                           Dropout(dropout),
                           GRU(int(hidden_dim/2), return_sequences=True), #activity_regularizer=regularizers.l1(10e-4)),
                           Dropout(dropout),
                           GRU(int(hidden_dim / 4), return_state=True)]#, activity_regularizer=regularizers.l1(10e-4))]

        # Wire up the layer list
        for i in range(1, len(encoder_layers)):
            encoder_layers[i] = encoder_layers[i](encoder_layers[i-1])

        # Grab the state of the final layer
        _, state_h = encoder_layers[-1]

        # Feedback outputs back into the decoder
        decoder_input = Input(shape=(None, self.out_shape[1]))

        # Desired encoder layer structure
        #encoder_layers = [Input(shape=(None, self.out_shape[1])),
        #                  GRU(hidden_dim/4, return_sequences=True),
        #                  GRU(int(hidden_dim / 2), return_sequences=True),
        #                  GRU(int(hidden_dim), return_sequences=True, return_state=True)]

        decoder_gru_1 = GRU(int(hidden_dim/4), return_sequences=True)
        decoder_gru_1a = decoder_gru_1(decoder_input, initial_state=state_h)

        decoder_gru_2 = GRU(int(hidden_dim/2), return_sequences=True)
        decoder_gru_2a = decoder_gru_2(decoder_gru_1a)

        decoder_gru_3 = GRU(int(hidden_dim), return_sequences=True, return_state=True)
        decoder_output, _ = decoder_gru_3(decoder_gru_2a)

        decoder_dense = Dense(self.out_shape[1], activation='linear')
        decoder_output = decoder_dense(decoder_output)

        # Full end-to-end model
        self.model = Model([encoder_layers[0], decoder_input], decoder_output)
        self.model.compile(optimizer=optimizer, loss=loss)

        # Just encoder model Input to output state
        self.encoder_model = Model(encoder_layers[0], state_h)

        ########
        # decoder_state_input = Input(shape=(int(hidden_dim / 4),))
        # decoder_output, state_c = decoder_gru(decoder_input, initial_state=decoder_state_input)
        # decoder_output = decoder_dense(decoder_output)
        ########

        decoder_state_input = Input(shape=(int(hidden_dim/4),))
        decoder_gru_1a = decoder_gru_1(decoder_input, initial_state=decoder_state_input)
        decoder_gru_2a = decoder_gru_2(decoder_gru_1a)
        decoder_output, state_c = decoder_gru_3(decoder_gru_2a)
        decoder_output = decoder_dense(decoder_output)

        self.decoder_model = Model([decoder_input, decoder_state_input], [decoder_output, state_c])

    def train(self, x, y, batch_size=10, epochs=50, verbose=2):
        yi = [np.concatenate([np.zeros((1, y.shape[2])), i[:-1, :]], axis=0) for i in y]
        yi = np.rollaxis(np.dstack(yi), -1)
        history = self.model.fit([x, yi], y,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=0.2,
                                 verbose=verbose,
                                 shuffle=True)

        plt.figure(figsize=(20, 5))
        plt.semilogy(history.history['loss'])
        plt.semilogy(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def predict_1(self, x):

        state_value = self.encoder_model.predict(x[np.newaxis, ...])

        target_seq = np.zeros(shape=(1, 1, self.out_shape[1]))
        res = []

        for _ in range(self.out_shape[0]):
            output_tokens, h = self.decoder_model.predict([target_seq, state_value])
            res.append(output_tokens[0, 0])
            target_seq = output_tokens
            state_value = h

        return np.concatenate([res])


class Seq2SeqModel:
    def __init__(self, hidden_dim, in_shape, out_shape, optimizer='RMSprop', loss='mse', dropout=0.5):

        self.hidden_dim = hidden_dim
        self.in_shape = in_shape
        self.out_shape = out_shape

        # encoder_input = Input(shape=(None, self.in_shape[1]))

        # g1 = GRU(hidden_dim, return_sequences=True)(encoder_input)
        # g2 = GRU(hidden_dim, return_sequences=True)(Dropout(dropout)(g1))

        encoder_layers = [Input(shape=(None, self.in_shape[1])),
                           GRU(hidden_dim, return_sequences=True),
                           Dropout(0.25),
                           GRU(int(hidden_dim/2), return_sequences=True),
                           Dropout(0.25)]

        for i in range(1, len(encoder_layers)):
            encoder_layers[i] = encoder_layers[i](encoder_layers[i-1])

        encoder = GRU(int(hidden_dim/4), return_state=True)
        _, state_h = encoder(encoder_layers[-1])

        decoder_input = Input(shape=(None, self.out_shape[1]))

        decoder_gru = GRU(int(hidden_dim/4), return_sequences=True, return_state=True)
        decoder_output, _ = decoder_gru(decoder_input, initial_state=state_h)

        decoder_dense = Dense(self.out_shape[1], activation='linear')
        decoder_output = decoder_dense(decoder_output)

        self.model = Model([encoder_layers[0], decoder_input], decoder_output)
        self.model.compile(optimizer=optimizer, loss=loss)

        self.encoder_model = Model(encoder_layers[0], state_h)

        decoder_state_input = Input(shape=(int(hidden_dim/4),))

        decoder_output, state_c = decoder_gru(decoder_input, initial_state=decoder_state_input)
        decoder_output = decoder_dense(decoder_output)

        self.decoder_model = Model([decoder_input, decoder_state_input], [decoder_output, state_c])

    def train(self, x, y, batch_size=10, epochs=50, verbose=2):
        yi = [np.concatenate([np.zeros((1, y.shape[2])), i[:-1, :]], axis=0) for i in y]
        yi = np.rollaxis(np.dstack(yi), -1)
        history = self.model.fit([x, yi], y,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=0.2,
                                 verbose=verbose,
                                 shuffle=True)

        plt.figure(figsize=(20, 5))
        plt.semilogy(history.history['loss'])
        plt.semilogy(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def predict_1(self, x):

        state_value = self.encoder_model.predict(x[np.newaxis, ...])

        target_seq = np.zeros(shape=(1, 1, self.out_shape[1]))
        res = []

        for _ in range(self.out_shape[0]):
            output_tokens, h = self.decoder_model.predict([target_seq, state_value])
            res.append(output_tokens[0, 0])
            target_seq = output_tokens
            state_value = h

        return np.concatenate([res])


class Seq2SeqModelBasic:
    def __init__(self, hidden_dim, in_shape, out_shape, optimizer='RMSprop', loss='mse'):

        self.hidden_dim = hidden_dim
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.encoder_input = Input(shape=(None, self.in_shape[1]), name='S2S_encoder_input')
        encoder = GRU(hidden_dim, return_state=True, name='S2S_encoder')
        _, state_h = encoder(self.encoder_input)

        decoder_input = Input(shape=(None, self.out_shape[1]), name='S2S_decoder_input')
        decoder_gru = GRU(hidden_dim, return_sequences=True, return_state=True)
        decoder_output, _ = decoder_gru(decoder_input, initial_state=state_h)
        decoder_dense = Dense(self.out_shape[1], activation='linear', name='S2S_dense_output')
        self.decoder_output = decoder_dense(decoder_output)

        self.model = Model([self.encoder_input, decoder_input], self.decoder_output, name='S2S_train_model')
        self.model.compile(optimizer=optimizer, loss=loss)

        self.encoder_model = Model(self.encoder_input, state_h, name="S2S_encoder")

        decoder_state_input = Input(shape=(self.hidden_dim,))

        decoder_output, state_c = decoder_gru(decoder_input, initial_state=decoder_state_input)
        decoder_output = decoder_dense(decoder_output)

        self.decoder_model = Model([decoder_input, decoder_state_input], [decoder_output, state_c], name='S2S_decoder')

    def train(self, x, y, batch_size=10, epochs=50, verbose=2):
        yi = [np.concatenate([np.zeros((1, y.shape[2])), i[:-1, :]], axis=0) for i in y]
        yi = np.rollaxis(np.dstack(yi), -1)
        history = self.model.fit([x, yi], y,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=0.2,
                                 verbose=verbose,
                                 shuffle=True)

        plt.figure(figsize=(20, 5))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def predict_1(self, x):

        state_value = self.encoder_model.predict(x[np.newaxis, ...])

        target_seq = np.zeros(shape=(1, 1, self.out_shape[1]))
        res = []

        for _ in range(self.out_shape[0]):
            output_tokens, h = self.decoder_model.predict([target_seq, state_value])
            res.append(output_tokens[0, 0])
            target_seq = output_tokens
            state_value = h

        return np.concatenate([res])
