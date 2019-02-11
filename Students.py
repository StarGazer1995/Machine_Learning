import os
import tensorflow as tf
import pandas as pd
import operator as op
import numpy as np
import pylab
import matplotlib.pyplot as plt
from scipy import misc
from tensorflow import keras
from PIL import Image

def get_tif(path):
    return[os.path.join(path,f) for f in os.listdir(path) if f.endswith('.tif')]

def images_show(image_name):
    plt.figure()
    plt.imshow(image_name)
    plt.colorbar()
    plt.show()

filename=os.path.abspath("G:\\Kaggle\\Data Set\\all\\train_labels.csv")
data=pd.read_csv(filename)
#train_dict={data.id,data.label}
train_images_list=get_tif("G:\\Kaggle\\Data Set\\all\\train")
train_images=[]
for i in range(0,5):
    train_images.append(np.array(misc.imread(train_images_list[i],mode='RGB')))

#train_images=train_images/255.0

input_data=tf.placeholder(tf.float32,shape=[None,96,96,3],name='input_data')
output=tf.placeholder(tf.int32,shape=[None,],name='output')
conv1=tf.layers.conv2d(
    input=input_data,
    filters=32,
    kernel_size=[5,5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
)
pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)

conv2=tf.layers.conv2d(
    input=pool1,
    filters=64,
    kernel_size=[5,5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
)
pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

conv3=tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

conv4=tf.layers.conv2d(
      inputs=pool3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

dense1 = tf.layers.dense(inputs=re1, 
                      units=1024, 
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dense2= tf.layers.dense(inputs=dense1, 
                      units=512, 
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
logits= tf.layers.dense(inputs=dense2, 
                        units=5, 
                        activation=None,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

loss=tf.losses.sparse_softmax_cross_entropy(labels=output,logits=logits)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), output)    
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

n_epoch=10
batch_size=64
sess=tf.InteractiveSession()  
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    start_time = time.time()
    
    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(train_images_list, data.label, batch_size, shuffle=True):
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
    print("   train loss: %f" % (train_loss/ n_batch))
    print("   train acc: %f" % (train_acc/ n_batch))
    
    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(train_images_list, data.label, batch_size, shuffle=False):
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1
    print("   validation loss: %f" % (val_loss/ n_batch))
    print("   validation acc: %f" % (val_acc/ n_batch))

sess.close()        
