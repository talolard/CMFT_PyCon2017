from .base import ModelBase
import tensorflow as tf
class LSTMBaseline(ModelBase):
    def __init__(self,args):
        super(LSTMBaseline, self).__init__(args)
    def get_logits(self):
        '''
        1) Fill this function. Use tf.nn.bidirectional_dynamic_rnn with an LSTM or GRU to generate logits for each point
        in the sequence.


        2) Bonus infer the sequence length using the sign trick (As done in ModelBase for the weights) to make the LSTM
            run faster.
        3) Time how long it takes for the model to process 100 batches. What loss do you get ?
        :return: logits = tf.contrib.layers.linear(result, num_outputs=128)
        '''

        pass

