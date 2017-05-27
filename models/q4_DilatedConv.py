from .base import ModelBase
import tensorflow as tf


class DilatedConv(ModelBase):
    def __init__(self, args):
        super(DilatedConv, self).__init__(args)

    def dilated_conv1d(self, input_):
        '''
            Write your own dilated_conv1d implementation such that you can pass the input into tensorflow dilated_conv2d function

        '''
    def get_logits(self):
        '''
        1) Stack the dilated conv1d you wrote above to get a receptive field of 128 characters
        2) Bonus: Concatenate the outputs of successive layers to form residual connections
        :return: logits = tf.contrib.layers.linear(result, num_outputs=128)
        '''

        pass

