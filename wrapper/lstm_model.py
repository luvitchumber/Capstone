#********
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential, load_model, save_model
from keras.layers import TimeDistributed, Dense, LSTM, Activation, RepeatVector, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import  to_categorical
import numpy as np


# steps depends on amount of data to be predicted
class LSTMModel:
	def __init__(self, x_train_shape, y_train_shape, intersection_list=None, num_epochs=50, l_rate=.001, batch_size=50):
		self.model = None
		self.models = []
		self.multi_model = False
		self.current_idx = 0
		self.input_length = x_train_shape
		self.steps = y_train_shape
		self.epochs = num_epochs
		self.l_rate = l_rate
		self.batch_size = batch_size
		if intersection_list:
			self.intersections=intersection_list
			self.multi_model = True
			self.load_network()
			

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

	def load_network(self, filepath=''):
		if self.multi_model:
			for i in self.intersections:
				#************
				self.models.append(tf.keras.models.load_model("./model/" + i + ".hdf"))
		else:
			#**********
			self.model = tf.keras.models.load_model(filepath, compile=True)
	
	def init_network(self, hidden_size, activation='relu', optimizer='adam', loss='mean_squared_error', verbose=False):
		model = Sequential()
		model.add(LSTM(units=hidden_size, return_sequences=True, input_shape=(self.input_length, 1)))
		model.add(LSTM(units=hidden_size, return_sequences=True))
		model.add(LSTM(units=hidden_size, return_sequences=True))
		model.add(LSTM(units=hidden_size))
		model.add(Dense(units=self.steps, activation=activation))
		model.compile(optimizer=optimizer, loss=loss)
		
		if verbose:
			print(model.summary())
		
		if self.multi_model:
			self.models.append(model)
		else:
			self.model = model


	def train(self, x_train, y_train, file_loc, validation_split=0.2, save=True):
		check_pointer = ModelCheckpoint(filepath=file_loc, verbose=1)
		callbacks_list = [check_pointer]
		if self.multi_model:
			print("Training is not possible with multi model mode!")
		else:
			self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
					   validation_split=validation_split, callbacks=callbacks_list)
		if save:
			print("Saving Trained Model")
			self.save_network(file_loc)

	def get_accuracy(self, x_data, y_data, verbose=False):
		if self.multi_model:
			try:
				test_output = self.model[self.intersections.index(x_data[0][0])].predict(x_data)
			except:
				print("Model for intersection does not exist!")
		else:
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
		print("Accuracy rate: {res}")
		return test_output
	
	def predict(self, x_data):
		test_output = 0
		if self.multi_model:
			try:
				print(self.intersections)
				print(str(x_data[0][0]))
				test_output = self.models[self.intersections.index(str(int(x_data[0][0][0])))].predict(x_data)
			except:
				print("Model for intersection does not exist!")
		else:
			test_output = self.model.predict(x_data)
			test_output = np.around(test_output[0], 0).astype(int)
		return test_output