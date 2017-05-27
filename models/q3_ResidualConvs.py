from .base import ModelBase
import tensorflow as tf


class ResidualConv(ModelBase):
    def __init__(self, args):
        super(ResidualConv, self).__init__(args)

    def conv1d(self, input_):
        '''
            Write your own conv1d implementation such that you can pass the input into tensorflow conv2d function
            Make sure the height of your filters makes sense
            What kind of padding should you use (VALID or SAME) For this task ?
            result = tf.nn.conv2d()

        '''
    def get_logits(self):
        '''
        1) Use the conv1d op you implemented to run a convolution over the input
        2) This time stack at least 20 layers one on top of the other
        3) Use residual connections to preserve gradients, that is, concatenate the input and output at each layer
        4) What is the receptive field of your model now?
        5) What is the size of the final representation of each "charecter" ?
        :return: logits = tf.contrib.layers.linear(result, num_outputs=128)
        '''

        pass

