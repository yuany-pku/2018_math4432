import tensorflow as tf
import numpy as np
import keras as ke
from tensorflow.examples.tutorials.mnist import input_data
from keras.callbacks import TensorBoard
from keras.datasets import mnist as mn
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Sequential


def weight_variabl(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def pretrain(x_train, num, shape, reuse = False, pre_module = Sequential(), reuse_num = 0, lay2_nb_filter = 0):
    autoencoder = Sequential()
    pre_autoencoder = Sequential()
    pre_weights = None
    if reuse:
        #for i in range(reuse_num):
        #hope to implement custom reuse layers and input_shape
        pre_autoencoder.add(Convolution2D(32, 3, 3, activation = 'relu', border_mode = 'same', 
                                      weights = pre_module.layers[0].get_weights(), input_shape = (28, 28, 1)))
        pre_autoencoder.add(MaxPooling2D((2, 2), border_mode = 'same'))
        x_train = pre_autoencoder.predict(x_train)
        pre_weights = pre_module.layers[2].get_weights()
      
    autoencoder.add(Convolution2D(lay2_nb_filter*2, 3, 3, activation = 'relu', border_mode = 'same', weights = pre_weights, input_shape = shape))
    autoencoder.add(MaxPooling2D((2, 2), border_mode = 'same'))
    autoencoder.add(Convolution2D(lay2_nb_filter, 3, 3, activation = 'relu', border_mode = 'same'))
    autoencoder.add(MaxPooling2D((2, 2), border_mode = 'same'))

    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Convolution2D(lay2_nb_filter*2, 3, 3, activation = 'relu', border_mode = 'same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Convolution2D(shape[2], 3, 3, activation = 'sigmoid', border_mode = ('valid' if reuse else 'same')))

    autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
    
    autoencoder.fit(x_train, x_train,
                    nb_epoch=200,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_train, x_train))
#                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
                   

    print("Pre_train of Convolution&pooling layer %d finished!"%num)
    return autoencoder
   


global sess
sess = tf.InteractiveSession()

#initialize the dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, shape = [-1, 28, 28, 1])

(x_train, _), (x_test, _) = mn.load_data('/home/LMX/mnist.pkl.gz')
x_train = x_train.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))

#pretrain&conv1,2
conv1 = pretrain(x_train, 1, (28, 28, 1), lay2_nb_filter = 16)
h_conv1 = Convolution2D(32, 3, 3, activation = 'relu', border_mode = 'same', weights = conv1.layers[0].get_weights())(x_image)
h_pool1 = MaxPooling2D((2, 2), border_mode = 'same')(h_conv1)
#print("ok1")
conv2 = pretrain(x_train, 2, (14, 14, 32), reuse = True, pre_module = conv1, reuse_num = 2, lay2_nb_filter = 8)
#print("ok2")
h_conv2 = Convolution2D(16, 3, 3, activation = 'relu', border_mode = 'same', weights = conv2.layers[0].get_weights())(h_pool1)
h_pool2 = MaxPooling2D((2, 2), border_mode = 'same')(h_conv2)

#Dense layer 
W_fc1 = weight_variabl([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_conv2, shape = [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Output
W_fc2 = weight_variabl([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Trainstep
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#evaluate
correct_prediction = tf.equal(tf.arg_max(y_conv,1), tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#init
init = tf.global_variables_initializer()
sess.run(init)

#train
print('Start to train on batch')
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if (i % 100 == 0):
        print("step %d : test accuracy %g"%(i, sess.run(accuracy, feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})))
    sess.run(train_step, feed_dict = {x:batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



