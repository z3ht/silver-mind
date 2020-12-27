from keras import models, layers
import numpy as np

class ChessNet:
  
  def __init__(self):
    self.model = None
    
  def train(self, X, y):
    """
    Train model with provided `X` and `y`
    """
    pass

  def save(self, save_path):
    """
    Save model to provided \`save_path\`
    """
    if self.model == None:
      raise RuntimeError("Model not trained; nothing to save")
    self.model.save(save_path, save_format="tf")

  def load(self, load_path):
    """
    Load model from provided `load_path`
    """
    self.model = models.load_model(load_path)
  
  def predict(self, serialized_board):
    if self.model is None:
      raise TypeError("Model has not been trained. Call \`self.train()\`")
    return (self.model.predict(np.array([serialized_board])) * 2) - 2

  def __call__(self, serialized_board):
    return self.predict(serialized_board)

class TwitchChess(ChessNet):
  
  def _create_model(self):
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=(5, 8, 8), data_format="channels_first", padding="same"))
    model.add(layers.Conv2D(16, (3, 3), activation="relu",  data_format="channels_first", padding="same"))
    model.add(layers.Conv2D(16, (3, 3), strides=(2,2), activation="relu",  data_format="channels_first", padding="same"))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", data_format="channels_first"))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", data_format="channels_first"))
    model.add(layers.Conv2D(32, (3, 3), strides=(2,2), activation="relu", padding="same", data_format="channels_first"))
    model.add(layers.Conv2D(64, (2, 2), activation="relu", padding="same", data_format="channels_first"))
    model.add(layers.Conv2D(64, (2, 2), activation="relu", padding="same", data_format="channels_first"))
    model.add(layers.Conv2D(64, (2, 2), strides=(2,2), activation="relu", padding="same", data_format="channels_first"))
    model.add(layers.Conv2D(128, (1, 1), activation="relu", padding="same", data_format="channels_first"))
    model.add(layers.Conv2D(128, (1, 1), activation="relu", padding="same", data_format="channels_first"))
    model.add(layers.Conv2D(128, (1, 1), activation="relu", padding="same", data_format="channels_first"))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["acc"])
    return model

  def train(self, X, y):
    model = self._create_model()
    
    model.summary()
    history = model.fit(X, y, batch_size=1024, epochs=20, validation_split=0.2, shuffle=True)

    self.model = model

    return history

class DeepChess(ChessNet):
    
  def _create_pos_to_vec(self):
    pass
    
  def _create_model(self):
    pass

  def train(self, X, y):
    model = self._create_model()
    
    model.summary()
    history = model.fit(X, y, batch_size=1024, epochs=20, validation_split=0.2, shuffle=True)

    self.model = model

    return history
        
    