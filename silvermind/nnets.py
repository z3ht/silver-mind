import keras

class ChessCNN:
	
	def __init__(self):
    self.model = None
  
  @staticmethod
  def _create_model():
    pass

  def train(self, X, y, test_split=False):
    """
    Train model with provided `X` and `y`

    Optional Parameter `test_split` for analyzing test split accuracy and loss
    """
    model = ChessCNN._create_model()
    model.fit(X, y, 

    self.model = model

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
    self.model = keras.models.load_model(load_path)

  def predict(self, serialized_board):
    if self.model is None:
      raise TypeError("Model has not been trained. Call \`self.train()\`")
    return (self.model.predict(serialized_board) * 2) - 2

  def __call__(self, serialized_board):
    return self.predict(serialized_board)

