from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, LSTM, Activation, RepeatVector, Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

# steps depends on amount of data to be predicted
class LstmModel:
  def __init__(self, num_epochs, lrate, batch_size):
    steps = y_train.shape[1]
    self.current_idx = 0
    #self.possible_classes = num_classes #list(range(0,1000))
    self.input_length = x_train.shape[1]
    self.model = None
    self.epochs = num_epochs
    self.lrate = lrate
    self.steps = steps
    self.batch_size = batch_size
    
  def get_batch(self, x_data, y_data):
    x = np.zeros((self.batch_size))
    y = np.zeros((self.batch_size))
    while True:
      if self.current_idx >= len(self.x_data):
      # reset the index back to the start of the data set
        self.current_idx = 0
      x = x_data[self.current_idx:self.current_idx + self.batch_size]
      y_tmp = y_data[self.current_idx :self.current_idx + self.batch_size]
      y = tf.keras.utils.to_categorical(temp_y, num_classes=self.possible_classes)
      # convert all of temp_y into a one hot representation
      self.current_idx += self.batch_size
    yield x, y

  def load_network(self, filepath):
    self.model = keras.models.load_model(filepath)

  #** can change the below to relu for activation
  def init_network(self, hidden_size, activation = 'softmax', bidirectional = False):
    self.model = Sequential()
    
    self.model.add(LSTM(units = hidden_size, return_sequences = True, input_shape = (self.input_length, 1)))
    #regressor.add(Dropout(0.2))

    self.model.add(LSTM(units = hidden_size, return_sequences = True))
    #regressor.add(Dropout(0.2))

    self.model.add(LSTM(units = hidden_size, return_sequences = True))
    #regressor.add(Dropout(0.2))

    self.model.add(LSTM(units = hidden_size))
    #regressor.add(Dropout(0.2))

    self.model.add(Dense(units = self.steps))
    
    #opt = Adam(lr =1e-2, decay=1e-2/self.epochs)
    #self.model.compile(optimizer = opt, loss = 'mean_squared_error')

    self.model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    print(self.model.summary())

  def train(self, x_train, y_train, filepath, filename, save = True, validation_split = 0.2): 
    checkpointer = ModelCheckpoint(filepath=filepath + '/' + filename + '.hdf5', verbose=1)
    callbacks_list = [checkpointer]
    self.model.fit(x_train, y_train, epochs = self.epochs, batch_size = self.batch_size, validation_split=validation_split, callbacks=callbacks_list)#, callbacks=[checkpointer])



  def predict(self, x_data, y_data, verbose=0):
    test_output = self.model.predict(x_data)
    test_output = np.around(test_output, 0).astype(int)
    corr = 0
    incorr = 0
    for pred, label in zip(test_output, y_data):
      pred = pred.astype(int)
      label = label.astype(int)
      for i, j in zip(pred, label):
        if  i == j:
          corr = corr +1
        else:
          incorr = incorr + 1
        if verbose == 1:
            print("Label:")
            print(label)
            print("Prediction:")
            print(pred)
    res = corr / (corr+incorr) * 100
    print("Accuracy rate: " + str(res))
    return test_output