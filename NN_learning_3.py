#coding:utf-8
import tensorflow as tf
import numpy as np

#学习率       学习指数衰减学习率

#设损失函数 loss=(w+1)^2
#使用指数衰减的学习率，在迭代初期得到较高的下降速度，可以在较小的训练轮数下取得更好的收敛度

LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_STEP = 1           #一般取值为：总样本数/BATCH_SIZE

#定义运行几轮的BENCH_SIZE计数器，初值为0，设置为不可训练
global_step = tf.Variable(0,trainable=False)
#定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase=True)

#定义待优化参数，初始值为5
w = tf.Variable(tf.constant(5,dtype=tf.float32))

#定义损失函数
loss = tf.square(w+1)

#定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step)

#生成会话，开始训练
step = 400
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(step):
        sess.run(train_step)
        learning_rate_var = sess.run(learning_rate)
        global_step_var = sess.run(global_step)
        w_var = sess.run(w)
        loss_var = sess.run(loss)
        if i % 20 ==0:
            print("step is :%d"%(i))
            print("learning_rate is :%f"%(learning_rate_var))
            print("global_step is : %f"%(global_step_var))
            print("w is : %f"%(w_var))
            print("loss is : %f "%(loss_var))
            print("")



