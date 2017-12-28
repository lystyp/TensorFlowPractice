import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import time
import numpy as np

# Create some variables.

# v1 = tf.get_variable("v1", shape = [1],initializer=tf.ones_initializer)

g1 = tf.Graph()
with g1.as_default():

    with tf.variable_scope("vc1"):
        v1 = tf.get_variable("v1", [1])
        init_op = tf.global_variables_initializer()

for i in range(10):
    asd = i

with tf.Session(graph=g1) as sess:
    print(asd)
    sess.run(init_op)
    
    with tf.variable_scope("vc1",reuse = True):
        vv1 = tf.get_variable("v1")
        print(sess.run(vv1))