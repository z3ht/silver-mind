from keras import models, layers
from keras import backend as K
import numpy as np
import os
import random


class ChessNet:

    def __init__(self):
        self.model = None
        self.state = None

    def train(self, X, y):
        """
        Train model with provided `X` and `y`
        """
        pass

    def save(self, save_path):
        """
        Save model to provided `save_path`
        """
        if self.model is None:
            raise RuntimeError("Model not trained; nothing to save")
        self.model.save(save_path, save_format="tf")

    def load(self, load_path):
        """
        Load model from provided `load_path`
        """
        self.model = models.load_model(load_path)

    def predict(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


class ValueNet(ChessNet):

    def predict(self, serialized_board):
        if self.model is None:
            raise TypeError("Model has not been trained. Call `self.train()`")
        return (self.model.predict(np.array([serialized_board])) * 2) - 2

    def __call__(self, target_board):
        return self.predict(target_board)


class ComparisonNet(ChessNet):

    def predict(self, serialized_board, best_board):
        if self.model is None:
            raise TypeError("Model has not been trained. Call `self.train()`")

        if best_board is None:
            raise TypeError("The DeepChess network requires `best_board`")

        return (self.model.predict(np.array([serialized_board, best_board])) * 2) - 2

    def __call__(self, target_board, best_board):
        return self.predict(target_board, best_board)


class TwitchChess(ValueNet):

    @staticmethod
    def _create_model():
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


class DeepChess(ComparisonNet):

    def __init__(self):
        super().__init__()
        self.dbn_layers = [320, 200, 100, 100]
        self.top = [100, 100, 2]
        self.dbn_batch_size = 512
        self.dbn_epochs = 5    # This could be much larger
        self.batch_size = 512
        self.epochs = 10

    @staticmethod
    def _random_samples(X, y, num_samples=800000):
        white_inds, black_inds = np.where(y == 1), np.where(y == 0)
        white_X, black_X = X[white_inds], X[black_inds]
        white_rows, black_rows = len(white_inds), len(black_inds)

        left, right = [], []
        ys = []
        for _ in range(num_samples):
            a = random.randint(0, white_rows - 1)
            b = random.randint(0, black_rows - 1)
            c = random.random()
            if c > 0.5:
                sample = [white_X[a], black_X[b], [1, 0]]
            else:
                sample = [black_X[b], white_X[a], [0, 1]]
            left.append(sample[0])
            right.append(sample[1])
            ys.append(sample[2])

        return [np.array(left), np.array(right)], np.array(ys)

    def _create_base_model(self, X):
        model = models.Sequential()
        for i in range(1, len(self.dbn_layers)):
            train_model = models.Sequential()
            train_model.add(layers.Dense(self.dbn_layers[i], activation="relu", input_dim=self.dbn_layers[i - 1]))
            train_model.add(layers.Dense(self.dbn_layers[i - 1], activation="relu"))
            train_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            train_model.fit(X, X, batch_size=self.dbn_batch_size, epochs=self.dbn_epochs)

            weights = train_model.layers[0].get_weights()
            X = K.function([train_model.input], [train_model.layers[0].output])([X])[0]     # Retrieve next X

            model.add(layers.Dense(self.dbn_layers[i], activation="relu", trainable=False, input_dim=self.dbn_layers[i-1]))
            model.layers[i-1].set_weights(weights)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _create_model(self, X, base_refresh=False, base_model_path="models/dbn.tf"):
        if base_refresh or not os.path.isfile(base_model_path):
            base = self._create_base_model(X)
            # Keras does not support saving a model created w/ another model running
            # base.save(base_model_path, save_format="tf")
        else:
            base = models.load_model(base_model_path)

        bot_left = layers.Input(shape=(self.dbn_layers[0],))
        bot_right = layers.Input(shape=(self.dbn_layers[0],))

        merged = layers.Concatenate()([base(bot_left), base(bot_right)])

        top1 = layers.Dense(self.top[0], activation="relu", input_dim=self.dbn_layers[-1])(merged)
        top2 = layers.Dense(self.top[1], activation="relu")(top1)
        top3 = layers.Dense(self.top[2], activation="softmax")(top2)

        model = models.Model(inputs=[bot_left, bot_right], outputs=top3)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["mae"])

        return model

    def train(self, X, y, base_refresh=False, base_model_path="models/dbn.tf", num_samples=800000):
        X = X.copy().reshape(X.shape[0], 8*8*5)

        validation_size = 100000

        model = self._create_model(X=X[:-validation_size], base_refresh=base_refresh, base_model_path=base_model_path)
        model.summary()

        X, y = self._random_samples(X=X, y=y, num_samples=num_samples)

        history = model.fit(
            [X[0][:-validation_size], X[1][:-validation_size]], y[:-validation_size],
            validation_data=([X[0][-validation_size:], X[1][-validation_size:]], y[-validation_size:]),
            use_multiprocessing=True, epochs=self.epochs,  batch_size=self.batch_size, shuffle=True
        )

        self.model = model

        return history

