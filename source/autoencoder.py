import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.regularizers import l1
from sklearn.base import TransformerMixin


class AutoEncoder(TransformerMixin):
    def __init__(self, add_noise=True):
        self.add_noise = add_noise

    def fit(self, x, y=None):
        """
        https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798
        https://blog.keras.io/building-autoencoders-in-keras.html

        :param x:
        :return:
        """
        input_size = x.shape[1]
        hidden_size = 128
        encoded_size = 64

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
        decoded = Dense(input_size, activation='tanh')(hidden_2)

        autoencoder = Model(input_img, decoded)

        # This model maps an input to its encoded representation
        self.encoder = Model(input_img, encoded)

        autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics='mean_squared_error')
        autoencoder.fit(x, x, epochs=100)

        return self

    def transform(self, x):
        return self.encoder.predict(x)
