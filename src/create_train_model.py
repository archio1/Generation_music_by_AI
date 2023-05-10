from keras.models import Sequential, save_model
from keras.layers import LSTM, Dropout, Dense


class MusicModel:

    def __init__(self, n_vocab):
        self.n_vocab = n_vocab
        self.model = None

    def create_and_train_model(self, network_input):
        # create the model
        self.model = Sequential()
        self.model.add(LSTM(256, return_sequences=True, input_shape=(network_input.shape[1], network_input.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(256))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(self.n_vocab, activation='softmax'))
        self.model.summary()

    def train_model(self, x_train, x_test, y_train, y_test):
        # compile the model using Adam optimizer
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # train the model on training sets and validate on testing sets
        self.model.fit(x_train, y_train, batch_size=128, epochs=80, validation_data=(x_test, y_test))

    def save_model(self, filename='s2s'):
        if self.model:
            save_model(self.model, filename)
        else:
            print("Model has not been created yet.")