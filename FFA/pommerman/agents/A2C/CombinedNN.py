import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten


class CombinedNN(keras.Model):
    def __init__(self, action_space):
        super().__init__()
        """
        self.layer1 = Dense(2048, activation="relu")
        self.layer2 = Dense(1024, activation="relu")
        self.layer3 = Dense(512, activation="relu")
        self.layer_out = Dense(6, activation="softmax")
        """

        self.conv_layer1 = Conv2D(32, (3,3), activation='relu', input_shape=(1, 11, 11, 12))
        self.conv_layer2 = Conv2D(64, (3,3), activation='relu')
        self.conv_layer3 = Conv2D(128, (3,3), activation='relu')
        self.conv_layer4 = Conv2D(256, (3,3), activation='relu')
        self.flatten = Flatten()
        self.dense_1 = Dense(500, activation='relu')
        self.dense_2 = Dense(100, activation='relu')
        self.dense_3 = Dense(50, activation='relu')

        self.out_policy = Dense(action_space, activation='softmax')
        self.out_value = Dense(1, activation=None)


    def call(self, input):
        out = self.conv_layer1(input)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.flatten(out)
        out = self.dense_1(out)
        out = self.dense_2(out)
        out = self.dense_3(out)

        policy = self.out_policy(out)
        value = self.out_value(out)

        return policy, value
