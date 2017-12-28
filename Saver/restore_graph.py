import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import time
import numpy as np

SAVE_FLAG = 0 # if 0 restore,else save

v1 = tf.Variable(tf.constant(2.2,shape=[1,1]), name = "v1")
v2 = tf.Variable(tf.constant(3.3,shape=[1,2]), name = "v1")


with tf.variable_scope("LSTM1"):
    cell = tf.contrib.rnn.BasicLSTMCell(1)
    init_state = cell.zero_state(1, dtype=tf.float32)
    (cell_output, state) = cell(v1, init_state)
    output1 = (cell_output, state)

with tf.variable_scope("LSTM2"):
    cell2 = tf.contrib.rnn.BasicLSTMCell(2)
    init_state2 = cell2.zero_state(1, dtype=tf.float32)
    (cell_output2, state2) = cell2(v2, init_state2)
    output2 = (cell_output2, state2)

init_op = tf.global_variables_initializer()

# saver = tf.train.import_meta_graph("tmp/model.ckpt.meta")

# #Way1
# # Way to restore many weight's in different model,cell.weights is a list
# # Saver(裡面貌似只能放一層的list，裡面不能再包list)
# #-----------------------------------------------------------
# l = []
# l.extend(cell.weights)
# l.extend(cell2.weights)
# saver = tf.train.Saver(l)


# with tf.Session() as sess:

#     sess.run(init_op)

#     if SAVE_FLAG == 0:
#         saver.restore(sess, "tmp/model.ckpt")
#     else:        
#         sess.run(init_op)
#         saver.save(sess, "tmp/model.ckpt")

#     print(sess.run(cell.weights))
#     print(sess.run(cell2.weights))
#     # for v in tf.all_variables():
#     #     print(v.name)
# #----------------------------------------------------------------

#Way2
# Way to restore many weight's in different model,cell.weights is a list
#-----------------------------------------------------------
saver = tf.train.Saver(cell.weights)
saver2 = tf.train.Saver(cell2.weights)
l = [saver, saver2]


with tf.Session() as sess:

    sess.run(init_op)

    if SAVE_FLAG == 0:
        for i in l:
            i.restore(sess, "tmp/model.ckpt")
    else:        
        sess.run(init_op)
        saver.save(sess, "tmp/model.ckpt")

    print(sess.run(cell.weights))
    print(sess.run(cell2.weights))
    # for v in tf.all_variables():
    #     print(v.name)
#----------------------------------------------------------------