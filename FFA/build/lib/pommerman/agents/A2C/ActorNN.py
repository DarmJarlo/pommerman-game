import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class ActorNN(keras.Model):
    def __init__(self, n_actions):
        super().__init__()
        self.layer1 = Dense(2048, activation="relu")
        self.layer2 = Dense(1024, activation="relu")
        self.layer3 = Dense(512, activation="relu")
        self.layer_out = Dense(n_actions, activation="softmax")

    def call(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer_out(out)

        return out
