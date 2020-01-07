import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer


class AddSingletonDepth(Layer):

    def __init__(self):
        super(AddSingletonDepth, self).__init__()

    def call(self, x, mask=None):
        x = tf.expand_dims(x, axis=-1)  # add a dimension of the right

        if tf.rank(x) == 4:
            return tf.transpose(x, (0, 3, 1, 2))
        else:
            return x


class Subtract(Layer):

    def __init__(self):
        super(Subtract, self).__init__()

    def call(self, x, mask=None):
        return x[0] - x[1]


class Slice(Layer):

    def __init__(self, selector):
        self.selector = selector
        super(Slice, self).__init__()

    def call(self, x, mask=None):

        selector = self.selector
        if len(self.selector) == 2 and not type(self.selector[1]) is slice and not type(self.selector[1]) is int:
            x = tf.transpose(x, [0, 2, 1])
            selector = (self.selector[1], self.selector[0])

        y = x[selector]

        if len(self.selector) == 2 and not type(self.selector[1]) is slice and not type(self.selector[1]) is int:
            y = tf.transpose(y, [0, 2, 1])

        return y


x = np.ones((10, 2, 3))
layer = AddSingletonDepth()
x = layer(x)
layer = Subtract()
x = layer(x)
print(x.shape)
