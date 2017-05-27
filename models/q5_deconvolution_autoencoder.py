import tensorflow as tf
'''
In this task you'll implemented a convolutional autoencoder for text. This is particularly useful for reducing a document
to a vector.
You'll need to implement an encoder function witch applies successive 1d residual convolutions and 1d pooling operations
 Then you'll implement a decoder which takes the resulting vector and expands it back to the size of the original sequence

'''
class DeconvAutoencoder():
    def __init__(self,args):
        self.args = args
        self.original = tf.placeholder(dtype=tf.int32,shape=[None,None])
        self.lower = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.embedded_source = self._embed_chars()
        logits = self.get_logits()
        self.loss_op = self._loss(logits)
        self.train_op = self._train(self.loss_op)
        self.predictions = self._prediction(logits)
    def get_logits(self):
        encoded = self.encoder(self.embedded_source)
        decoded = self.decoder(encoded)
        logits = tf.contrib.layers.linear(decoded, num_outputs=128)
        return logits
    def avg_pool1d(self,input_):
        '''
        Implement 1d average pooling by wrapping tensorflows avg_pooling function.
        :return:
        '''
    def conv_1d_transpose(self,input_,**kwargs):
        '''
        Implement 1d "deconvolution" by wrapping tensorflows conv2d_transpose function.
        :return:
        '''

    def residual_conv_layer_1d(self,input_,**kwargs):
        '''
        Implement a series of stacked 1d convolutions with residual connections (That is concatenate the input and output
        at each layer)
        :return:
        '''
    def encoder(self,input_):
        '''
        Implement DenseNet (more or less). Use successive residual and pooling layers to reduce a sentance to a vector
         representation. Assume the input length is padded/truncated to 512 charecters
        :return:
        '''
    def decoder(self,input_):
        '''
        Implement DenseNet  (this time backward). Use successive residual and deconvolution layers to expand a vector
        to a sentence
        :return:
        '''


    def _embed_chars(self):
        embedding_size = 4
        embedding_matrix = tf.get_variable("embedding_matrix", shape=[128, embedding_size],
                                           dtype=tf.float32)

        return tf.nn.embedding_lookup(embedding_matrix,self.original)



    def _loss(self,logits):
        lengths = tf.reduce_sum(tf.sign(self.original),axis=1)
        maxlen = self.original.get_shape().as_list()[1]
        mask = tf.sequence_mask(lengths,dtype=tf.float32,maxlen=500)
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

