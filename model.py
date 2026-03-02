from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, SpatialDropout1D
from tensorflow.keras.models import Sequential

def create_model(vocab_size, maxlen=25, embedding_dim=16):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(16, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    model = create_model(vocab_size=3000)
    model.summary()
