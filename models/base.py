import tensorflow as tf
import abc

class ModelBase(metaclass=abc.ABCMeta):
    def __init__(self,args):
        self.args = args
        self.original = tf.placeholder(dtype=tf.int32,shape=[None,None])
        self.lower = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.batch_max_len = self.original.get_shape().as_list()[1]
        self.embedded_source = self._embed_chars()
        logits = self.get_logits()
        self.loss_op = self._loss(logits)
        self.train_op = self._train(self.loss_op)
        self.predictions = self._prediction(logits)

    @abc.abstractmethod
    def get_logits(self):
        pass

    def _embed_chars(self):
        embedding_size = 4
        embedding_matrix = tf.get_variable("embedding_matrix", shape=[128, embedding_size],
                                           dtype=tf.float32)

        return tf.nn.embedding_lookup(embedding_matrix,self.lower)



    def _loss(self,logits):
        lengths = tf.reduce_sum(tf.sign(self.original),axis=1)
        max_len =self.original.get_shape().as_list()[1]
        mask = tf.sequence_mask(lengths,dtype=tf.float32,maxlen=max_len)
        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                targets=self.original,
                                                weights = mask
                                                )
        return loss

    def _train(self,loss):
        #lr = tf.train.exponential_decay(learning_rate=self.args.lr)
        lr = self.args.lr
        opt = tf.train.AdamOptimizer(lr)
        return opt.minimize(loss)

    def _prediction(self,logits):
        sm = tf.nn.softmax(logits)
        preds = tf.argmax(sm,axis=2)
        return preds

