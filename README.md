## Deep Seq2Seq Model 
### Implimented in Keras. 

This is an extension of the model described on 
https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html.

It allows for multiple RNN layers in both the encoder and decoder networks. The implementation
has been written with time-series modelling in mind so some tweaks may be needed for other 
applications such as NLP.

The tricky part of implementing the seq2seq with multiple recurrent layers arises from
the different modes of behaviour in the training and prediction phases. 

I've overcome this by copying the decoder network at prediction time with the appropriate changes
changes made to allow for the feedback of states and update steps (this would be 
somewhat easier if there was the ability to update flags like 'stateful'). 