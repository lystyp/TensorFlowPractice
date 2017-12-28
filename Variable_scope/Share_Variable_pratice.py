import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import time
import numpy as np


initializer = tf.random_uniform_initializer(-2, 2)
v1 = tf.Variable(tf.constant(2.2,shape=[1,1]), name = "v1")




with tf.variable_scope("Model", reuse=None, initializer=initializer):
     cell1 = tf.contrib.rnn.BasicLSTMCell(1)
     initial_state1 = cell1.zero_state(1, tf.float32)
     (cell_output1, state1) = cell1(v1, initial_state1)
with tf.variable_scope("Model", reuse=True, initializer=initializer):
     cell2 = tf.contrib.rnn.BasicLSTMCell(1)
     initial_state2 = cell2.zero_state(1, tf.float32)
     (cell_output2, state2) = cell2(v1, initial_state2)
with tf.variable_scope("Model2", reuse=None, initializer=initializer):
     cell3 = tf.contrib.rnn.BasicLSTMCell(1)
     initial_state3 = cell3.zero_state(1, tf.float32)
     (cell_output3, state3) = cell3(v1, initial_state3)
init_op = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init_op)

    print("-------------------------------------------------------------")
    print(sess.run(cell1.weights))
    print("-------------------------------------------------------------")
    print(sess.run(cell2.weights))
    print("-------------------------------------------------------------")
    print(sess.run(cell3.weights))
