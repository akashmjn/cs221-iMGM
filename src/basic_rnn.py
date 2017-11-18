import numpy as np
import tensorflow as tf

###############################################################################

class RNNMusic:
    def add_placeholders(self):
        self.inputs = tf.placeholder(shape=(None,5,128), dtype=tf.int32)
        self.labels = tf.placeholder(shape=(None,128), dtype=tf.int32)
    
    def create_feed_dict(self, inputs, labels=None):
        feed_dict = {}
        feed_dict[self.inputs] = inputs
        if not labels is None:
            feed_dict[self.labels] = labels
        return feed_dict
    
    