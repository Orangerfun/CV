import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy as np
from PIL import Image
# import train_cnn
np.set_printoptions(threshold=np.inf,precision=1)

mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
batch_size=100
n_batch=mnist.train.num_examples//batch_size

def change2value(train_data):
    #将图片像素都化成0and1,黑色地方数值为零，白色为1
    average=np.average(train_data)
    train_data=(train_data<average)+0
    return train_data

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")  #[input,filter,strides,padding]

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1], padding="SAME")      #input.shape[batchsize,height,width,channels] ksize.shape[1,heiht,width,1]池化窗口大小，strides=[1,stride,stride,1]每个维度步长

x=tf.placeholder(tf.float32,[None,784],name="x_input")   #[batchsize,28*28]
y=tf.placeholder(tf.float32,[None,10],name="y_input")       #[batchsize,n_classes]
x_image=tf.reshape(x,[-1,28,28,1],"x_image")        #[batchsize,height,width,channels]

w_conv1=weight_variable([5,5,1,32])     #filter[height,width,n_channels,filter_num]            
b_conv1=bias_variable([32])  #32个卷积层，每个卷积层bias都为0.1
#conv1
h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
#maxpool1
h_pool1=max_pool_2x2(h_conv1)  

w_conv2=weight_variable([5,5,32,64])  
b_conv2=bias_variable([64]) 
#conv2
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
#maxpool2
h_pool2=max_pool_2x2(h_conv2)

w_fc1=weight_variable([7*7*64,1024]) 
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
wx_plus_b1=tf.matmul(h_pool2_flat,w_fc1)+b_fc1   
h_fc1=tf.nn.relu(wx_plus_b1)

keep_prob=tf.placeholder(tf.float32,name="keep_prob")
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob,name="keep_prob")

w_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

wx_plus_b=tf.matmul(h_fc1_drop,w_fc2)+b_fc2
prediction=tf.nn.softmax(wx_plus_b)

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction),name="loss")
tf.summary.scalar("loss",cross_entropy)
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

prediction_num=tf.argmax(prediction,1)
saver=tf.train.Saver()

def main(filename):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,"./savepoint/cnn.ckpt")
        img=Image.open(filename)
        img=np.array(img.convert("L"))
        image=change2value(img).reshape(-1,784)
        result=sess.run(prediction_num,feed_dict={x:image,keep_prob:1.0})[0]
    return result
