import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model

import numpy as np

    
class ConvLayer(Layer):
    def __init__(self, filters, kernel_size, padding:str='same'):
        super(ConvLayer, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.conv = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.pool = layers.MaxPool2D((2,2))
        
    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

    
class Prototypical_Network(Model):
    def __init__(self, w:int=28, h:int=28, c:int=1):
        super(Prototypical_Network, self).__init__()
        self.w, self.h, self.c = w, h, c

        self.encoder = tf.keras.Sequential([
            ConvLayer(64, 3, 'same'),
            ConvLayer(64, 3, 'same'),
            ConvLayer(64, 3, 'same'),
            ConvLayer(64, 3, 'same'),
            layers.Flatten()
        ])
        
        
    def call(self, support, query):
        n_way = support.shape[0]
        n_support = support.shape[1]
        n_query = query.shape[1]
        
        reshaped_s = tf.reshape(support, (n_way * n_support, self.w, self.h,self.c))
        reshaped_q = tf.reshape(query, (n_way * n_query, self.w, self.h,self.c))
        
        # Embeddings are in the shape of (n_support+n_query, 64)
        
        embeddings = self.encoder(tf.concat([reshaped_s, reshaped_q], axis=0))
     
        # Support prototypes are in the shape of (n_way, n_support, 64)
        s_prototypes = tf.reshape(embeddings[:n_way * n_support], [n_way, n_support, embeddings.shape[-1]])
        # Find the average of prototypes for each class in n_way
       
        s_prototypes = tf.math.reduce_mean(s_prototypes, axis=1)
    
   
        # Query embeddings are the remainding embeddings
        q_embeddings = embeddings[n_way * n_support:]
        
        
        loss = 0.0
        acc = 0.0
        ############### Your code here ###################
            # TODO: finish implementing this method.
            # For a given task, calculate the Euclidean distance
            # for each query embedding and support prototypes.
            # Then, use these distances to calculate
            # both the loss and the accuracy of the model.
            # HINT: you can use tf.nn.log_softmax()

        y = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query))
        y_onehot = tf.cast(tf.one_hot(y, n_way), tf.float32)

        s_prototypes = tf.tile(s_prototypes,[q_embeddings.shape[0],1])
        q_embeddings = tf.repeat(q_embeddings, n_way, axis=0)
        
        dist = tf.math.reduce_sum(tf.square(s_prototypes-q_embeddings),axis=1)
        dist = tf.reshape(dist, (-1,n_way))


        log_p_y = tf.nn.log_softmax(-dist, axis=-1)
        log_p_y = tf.reshape(log_p_y, [n_way, n_query, -1])
       
        
        loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_onehot, log_p_y), axis=-1), [-1]))
        
        eq = tf.cast(tf.equal(
            tf.cast(tf.argmax(log_p_y, axis=-1), tf.int32), 
            tf.cast(y, tf.int32)), tf.float32)
        acc = tf.reduce_mean(eq)
        
        ##################################################
        
        return loss, acc
    
    def dist(self, x, y):
        
        n = x.shape[0]
        d = x.shape[1]
        m = y.shape[0]
        
        t1 = tf.reshape(x, (n, 1, d))
        t2 = tf.reshape(y, (1, m, d))
        
        return tf.norm(t1-t2, ord='euclidean', axis=2)
    
    
    