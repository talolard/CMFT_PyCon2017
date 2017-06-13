import tensorflow as tf
import sys
from arg_parse import parse_args
from models.small_conv import SmallConv as Model
from prepare_data import DataLoader
import re
import numpy as np
def main(args):
    args = parse_args(args)
    DL = DataLoader(args)
    print("Loading data")
    DL.load_data()
    print("Done")


    with tf.Session() as sess:
        print("Loading model")
        model = Model(args)
        init = tf.global_variables_initializer()
        print("Number of trainable vars is {}".format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        sess.run(init)
        for epoch in range(10):
            for num,batch in enumerate(DL.get_batch()):
                orig,lower = batch
                feed_dict = {model.original:orig,
                             model.lower:lower,

                             }
                fetches = [model.train_op,model.loss_op,model.predictions]
                _,loss,preds = sess.run(fetches=fetches,feed_dict=feed_dict)
                if num% 25 ==0:
                    print(loss)
                if num %100 ==0:
                    print('************************')
                    print(DL.ar_to_str(orig[0]))
                    print(DL.ar_to_str(preds[0]))
                    print(DL.ar_to_str(lower[0]))
if __name__ =='__main__':
    main(sys.argv[1:])

