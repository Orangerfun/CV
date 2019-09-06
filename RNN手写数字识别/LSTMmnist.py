import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("./data",one_hot=True)

n_input=28      #每个时序的长度
n_steps=28      #时序个数
n_hidden=128        #rnn输出维度
n_classes=10        #类别数


tf.reset_default_graph()
x=tf.placeholder("float",[None,n_steps,n_input])       #[batch_size,n_steps,n_input]
y=tf.placeholder('float',[None,n_classes])

x1=tf.unstack(x,n_steps,1)   #详见D:\Desktop\Admin\LearningFile\Note\TF

lstm_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1)

#静态单层RNN网络,输入必须是列表形式
#outputs,states=tf.contrib.rnn.static_rnn(lstm_cell,x1,dtype=tf.float32)   #outputs.shape[batchsize,n_steps,n_hidden]   [None,28,128]

#动态单层RNN网络，输入为张量
# outputs,_=tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32)
# outputs=tf.transpose(outputs,[1,0,2])   #[batchsize,max_step,input_size]---->[max_step,batchsize,inputsize]这样取最后时间序列得到就是【batchsize,outputsize]

#静态多层网络
# stacked_rnn=[]
# for i in range(3):
#     stacked_rnn.append(tf.contrib.rnn.LSTMCell(n_hidden))
# mcell=tf.contrib.rnn.MultiRNNCell(stacked_rnn)
# outputs,_=tf.contrib.rnn.static_rnn(mcell,x1,dtype=tf.float32)

#动态多层网络
gru=tf.contrib.rnn.GRUCell(n_hidden)
lstm_cell=tf.contrib.rnn.LSTMCell(n_hidden)
mcell=tf.contrib.rnn.MultiRNNCell([gru,lstm_cell])
outputs,states=tf.nn.dynamic_rnn(mcell,x,dtype=tf.float32)
outputs=tf.transpose(outputs,[1,0,2])

pred=tf.contrib.layers.fully_connected(outputs[-1],n_classes)

learning_rate=0.001
training_iters=100000
batch_size=128
display_step=10

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

saver=tf.train.Saver(max_to_keep=1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step=1
    while step*batch_size<training_iters:
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        batch_x=batch_x.reshape(batch_size,n_steps,n_input)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if step%display_step==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
            print("iter:"+str(step*batch_size)+","+"Minibatch loss:{:.6f}".format(loss)+",Training_accuracy:{:.5f}".format(acc))
        step+=1
        saver.save(sess,"./multistatic_Lstm.cpkt",global_step=step)
    print("Finished...")
    # saver.restore(sess,"./Lstm.cpkt-782")

    test_len=128
    test_data=mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
    test_label=mnist.test.labels[:test_len]
    print("Testing Accuracy:",sess.run(accuracy,feed_dict={x:test_data,y:test_label}))

###statci one:   96
###dynamic one:  91
###static mul: 97
###dynamic mul: 1.0




