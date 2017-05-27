from .base import ModelBase
import tensorflow as tf


class SimpleConv(ModelBase):
    def __init__(self, args):
        super(SimpleConv, self).__init__(args)

    def conv1d(self,input_):
        '''
            Write your own conv1d implementation such that you can pass the input into tensorflow conv2d function
            Make sure the height of your filters makes sense
            What kind of padding should you use (VALID or SAME) For this task ?
            result = tf.nn.conv2d()

        '''


    def get_logits(self):
        '''
        1) Use the conv1d op you implemented to run a convolution over the input
        2) Don't forget to use an activation function such as tf.nn.relu
        3) Bonus if you stack a few conv layers one atop the other
        4) What is the receptive field of your output layer ?
        :return: logits = tf.contrib.layers.linear(result, num_outputs=128)
        '''

        pass

