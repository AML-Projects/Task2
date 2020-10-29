import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.regularizers import l1
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from helpers.visualize import visualize_true_labels
from logcreator.logcreator import Logcreator


class AutoEncoder(TransformerMixin):
    def __init__(self, add_noise=False, encoded_size=64, scaling_on=True):
        Logcreator.info("\nAuto encoder:")
        self.add_noise = add_noise
        self.encoded_size = encoded_size
        self.scaling_on = scaling_on
        Logcreator.info("add_noise =", add_noise)
        Logcreator.info("encoded size =", encoded_size)
        Logcreator.info("scaling_on =", scaling_on)

    def scale_input(self, x):
        if self.scaling_on:
            scaler = MinMaxScaler(feature_range=(0, 1))
            x = scaler.fit_transform(x)

        return x

    def fit(self, x, y=None):
        """
        https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798
        https://blog.keras.io/building-autoencoders-in-keras.html

        :param x:
        :return:
        """
        input_size = x.shape[1]
        hidden_size = 128
        encoded_size = self.encoded_size

        x = self.scale_input(x)

        # add noise to input
        if self.add_noise:
            noise_factor = 0.3
            x_train_noisy = x + noise_factor * np.random.normal(size=x.shape)
            x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
            x = x_train_noisy

        input_img = Input(shape=(input_size,))
        hidden_1 = Dense(hidden_size, activation='relu')(input_img)
        encoded = Dense(encoded_size, activation='relu', activity_regularizer=l1(10e-6))(hidden_1)
        hidden_2 = Dense(hidden_size, activation='relu')(encoded)
        decoded = Dense(input_size, activation='sigmoid')(hidden_2)

        autoencoder = Model(input_img, decoded)

        # This model maps an input to its encoded representation
        self.encoder = Model(input_img, encoded)

        cross_entropy_loss_on = False
        if cross_entropy_loss_on:
            autoencoder.compile(optimizer='nadam', loss='binary_crossentropy', metrics='mean_absolute_error')
        else:
            autoencoder.compile(optimizer='nadam', loss='mean_absolute_error', metrics='binary_crossentropy')

        autoencoder.fit(x, x, epochs=200)

        visualize_true_labels(x, y, "Auto Encoded")

        return self

    def transform(self, x):
        x = self.scale_input(x)
        return self.encoder.predict(x)
