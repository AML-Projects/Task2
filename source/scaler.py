"""
Data normalizer
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Sarah Morillo'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, sleonardo@student.ethz.ch"

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from logcreator.logcreator import Logcreator


class Scaler:
    def __init__(self, name):
        self.name = name
        Logcreator.info("Start normalizer")

    def transform_custom(self, x_train, y_train, x_test):
        switcher = {
            'stdscaler': self.standard_scaler,
            'minmaxscaler': self.minmax_scaler,
            'robustscaler': self.robust_scaler
        }
        norm = switcher.get(self.name)

        return norm(x_train, y_train, x_test)

    def standard_scaler(self, x_train, y_train, x_test):
        return self.scale_data(StandardScaler(), x_train, y_train, x_test)

    def minmax_scaler(self, x_train, y_train, x_test):
        return self.scale_data(MinMaxScaler(), x_train, y_train, x_test)

    def robust_scaler(self, x_train, y_train, x_test):
        return self.scale_data(RobustScaler(), x_train, y_train, x_test)

    def scale_data(self, scaler, x_train, y_train, x_test):
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        Logcreator.info("\ntrain shape", x_train.shape)

        return x_train, y_train, x_test
