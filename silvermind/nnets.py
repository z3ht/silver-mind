from keras import models, layers

class ChessCNN:
  
  def __init__(self):
    self.model = None
  
  @staticmethod
  def _create_model():
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
    """
    Train model with provided `X` and `y`
    """
    model = ChessCNN._create_model()
    model.summary()
    history = model.fit(X, y, batch_size=128, epochs=100, validation_split=0.2, shuffle=True)

    self.model = model

    return history

  def save(self, save_path):
    """
    Save model to provided \`save_path\`
    """
    if self.model == None:
      raise RuntimeError("Model not trained; nothing to save")
    self.model.save(save_path)

  def load(self, load_path):
    """
    Load model from provided \`load_path\`
    """
    self.model = models.load_model(load_path)

  def predict(self, serialized_board):
    if self.model is None:
      raise TypeError("Model has not been trained. Call \`self.train()\`")
    return (self.model.predict(serialized_board) * 2) - 2

  def __call__(self, serialized_board):
    return self.predict(serialized_board)

