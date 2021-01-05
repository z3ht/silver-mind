from keras import models, layers
from keras import backend as K
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
            raise TypeError("Model has not been trained. Call `self.train()`")
        return (self.model.predict(np.array([serialized_board])) * 2) - 2

    def __call__(self, serialized_board):
        return self.predict(serialized_board)


class TwitchChess(ChessNet):

    def _create_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=(5, 8, 8), data_format="channels_first",
                                padding="same"))
        model.add(layers.Conv2D(16, (3, 3), activation="relu", data_format="channels_first", padding="same"))
        model.add(
            layers.Conv2D(16, (3, 3), strides=(2, 2), activation="relu", data_format="channels_first", padding="same"))
        model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", data_format="channels_first"))
        model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", data_format="channels_first"))
        model.add(
            layers.Conv2D(32, (3, 3), strides=(2, 2), activation="relu", padding="same", data_format="channels_first"))
        model.add(layers.Conv2D(64, (2, 2), activation="relu", padding="same", data_format="channels_first"))
        model.add(layers.Conv2D(64, (2, 2), activation="relu", padding="same", data_format="channels_first"))
        model.add(
            layers.Conv2D(64, (2, 2), strides=(2, 2), activation="relu", padding="same", data_format="channels_first"))
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
        history = model.fit(X, y, batch_size=512, epochs=20, validation_split=0.2, shuffle=True)

        self.model = model

        return history


class DeepChess(ChessNet):

    def __init__(self):
        super().__init__()
        self.dbn_layers = [773, 100, 100, 100]
        self.batch_size = 256
        self.epochs = 2000

    def _create_dbn_model(self, X, y):
        model = models.Sequential()
        for i in range(1, len(self.dbn_layers)):
            train_model = models.Sequential()
            train_model.add(layers.Dense(self.dbn_layers[i], activation="relu", input_dim=self.dbn_layers[i - 1]))
            train_model.add(layers.Dense(self.dbn_layers[i - 1], activation="relu"))
            train_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
            train_model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)

            X = K.function([train_model.input], [train_model.layers[0].output])([X])[0]     # Retrieve next X
            weights = train_model.layers[0].get_weights()

            model.add(layers.Dense(self.dbn_layers[i], activation="relu", trainable=False, input_dim=self.dbn_layers[i-1]))
            model.layers[i-1].set_weights(weights)
        return model

    def _create_model(self, X, y):
        base = self._create_dbn_model(X, y)

        bot_left = layers.Input(shape=(self.dbn_layers[0],))
        bot_right = layers.Input(shape=(self.dbn_layers[0],))

        # Base layers are frozen so reuse is fine
        merged = layers.Concatenate([base(bot_left), base(bot_right)])

        top2 = layers.Dense(100, activation="relu")(merged)
        top3 = layers.Dense(100, activation="relu")(top2)
        top4 = layers.Dense(2, activation="softmax")(top3)

        model = models.Model(inputs=[bot_left, bot_right], outputs=top4)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        return model

    def train(self, X, y):
        model = self._create_model(X, y)

        model.summary()
        history = model.fit(X, y, batch_size=512, epochs=200, validation_split=0.2, shuffle=True)

        self.model = model

        return history
