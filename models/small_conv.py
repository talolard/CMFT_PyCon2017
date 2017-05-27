from .base import ModelBase
import tensorflow as tf
from .densenet import ops
class SmallConv(ModelBase):
    def __init__(self,args):
        super(SmallConv, self).__init__(args)
    def get_logits(self):
        encoded = ops.makeBlock(self.embedded_source, growth_rate=16, num_layers=30)
        logits = tf.contrib.layers.linear(encoded, num_outputs=128)
        return logits

