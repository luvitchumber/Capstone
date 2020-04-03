from keras.models import Sequential, load_model, save_model
from keras.layers import TimeDistributed, Dense, LSTM, Activation, RepeatVector, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import  to_categorical
import numpy as np


# steps depends on amount of data to be predicted
class LSTMModel:
    def __init__(self, x_train_shape, y_train_shape, num_epochs=50, l_rate=.001, batch_size=50):
        self.model = None
        self.current_idx = 0
        self.input_length = x_train_shape
        self.steps = y_train_shape
        self.epochs = num_epochs
        self.l_rate = l_rate
        self.batch_size = batch_size

    def get_batch(self, x_data, y_data):
        x = np.zeros(self.batch_size)
        y = np.zeros(self.batch_size)
        while True:
            if self.current_idx >= len(self.x_data):
                # reset the index back to the start of the data set
                self.current_idx = 0
            x = x_data[self.current_idx:self.current_idx + self.batch_size]
            y_tmp = y_data[self.current_idx:self.current_idx + self.batch_size]
            y = to_categorical(y_tmp, num_classes=self.possible_classes)
            # convert all of temp_y into a one hot representation
            self.current_idx += self.batch_size
            yield x, y

    def save_network(self, filepath):
        save_model(self.model, filepath, overwrite=True, include_optimizer=True)

    def load_network(self, filepath):
		self.model = keras.models.load_model(filepath)
	
	
    def init_network(self, hidden_size, activation='linear', optimizer='adam', loss='mean_squared_error', verbose=False):
        self.model = Sequential()

        self.model.add(LSTM(units=hidden_size, return_sequences=True, input_shape=(self.input_length, 1)))

        self.model.add(LSTM(units=hidden_size, return_sequences=True))

        self.model.add(LSTM(units=hidden_size, return_sequences=True))

        self.model.add(LSTM(units=hidden_size))

        self.model.add(Dense(units=self.steps, activation=activation))

        self.model.compile(optimizer=optimizer, loss=loss)
        if verbose:
            print(self.model.summary())

    def train(self, x_train, y_train, file_loc, validation_split=0.2, save=True):
        check_pointer = ModelCheckpoint(filepath=file_loc, verbose=1)
        callbacks_list = [check_pointer]
        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                       validation_split=validation_split, callbacks=callbacks_list)
        if save:
            print("Saving Trained Model")
            self.save_network(file_loc)

    def predict(self, x_data, y_data, verbose=False):
        test_output = self.model.predict(x_data)
        test_output = np.around(test_output, 0).astype(int)
        corr = 0
        incorr = 0
        for pred, label in zip(test_output, y_data):
            pred = pred.astype(int)
            label = label.astype(int)
            for i, j in zip(pred, label):
                if i == j:
                    corr = corr + 1
                else:
                    incorr = incorr + 1
                if verbose == 1:
                    print("Label:", label)
                    print("Prediction:", pred)

        res = corr / (corr + incorr) * 100
        print(f"Accuracy rate: {res}")
        return test_output
