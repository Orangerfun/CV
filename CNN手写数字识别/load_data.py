from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)			#one_hot=True表示将标签转换成one_hot编码
print(mnist.train.labels)
print(mnist.train.labels.shape)
print(mnist.test.images)
#打印MNIST信息
print("data.shape:",mnist.train.images.shape)
im=mnist.train.images[1]
im=im.reshape(-1,28)
plt.imshow(im)
plt.show()