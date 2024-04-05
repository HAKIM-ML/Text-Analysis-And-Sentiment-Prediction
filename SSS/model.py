from tensorflow.keras.layers import (LSTM, BatchNormalization, Dense, Dropout,
                                     Embedding)
from tensorflow.keras.models import Sequential


class SentimentModel:
    def __init__(self, vocab_size, embedding_dim, max_length):
        self.vocab_size = vocab_size
        self.embdding_dim = embedding_dim
        self.max_length = max_length
        
    def create_model(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size, self.embdding_dim))
        model.add(LSTM(32, return_sequences=True))
        model.add(BatchNormalization())
        
        model.add(LSTM(64))
        model.add(BatchNormalization())
        
        model.add(Dense(64, activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation = 'softmax'))
        
        model.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam',
                      metrics = ['accuracy'])
        return model