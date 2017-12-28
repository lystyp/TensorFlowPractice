import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import time
import numpy as np

SAVE_FLAG = 0 # if 0 restore,else save

v1 = tf.Variable(tf.constant(2.2,shape=[1,1]), name = "v1")
v2 = tf.Variable(tf.constant(3.3,shape=[1,2]), name = "v2")

if SAVE_FLAG == 1:
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



with tf.Session() as sess:

    # save and restore weight in LSTM 
    if SAVE_FLAG == 0:
        print("Load~~~")
        print("load graph")
        with tf.gfile.FastGFile("tmp/test.pb",'rb') as f:  # 讀pb檔 !!! 'rb'應該是指讀binary檔，記得之前存graph要存binary這邊才不會error
            graph_def = tf.GraphDef()                   # 建預設的graph
            graph_def.ParseFromString(f.read())         # 用預設的graph讀檔
            sess.graph.as_default()                     # 在這個context底下的graph把他給設成預設(初始化的意思?)
            tf.import_graph_def(graph_def, name='')     # 把剛剛讀的graph塞到目前的context底下
        saver = tf.train.Saver()
        saver.restore(sess, "tmp/model.ckpt")
        print(tf.global_variables())
    else:   
        print("Save~~~")     
        print("Save graph path >>> " )
        print( tf.train.write_graph(sess.graph_def, "tmp", "test.pb", False)) 
        sess.run(init_op)
        saver = tf.train.Saver()
        saver.save(sess, "tmp/model.ckpt")
    if SAVE_FLAG == 1 : 
        print(sess.run(cell.weights))
        print(sess.run(cell2.weights))
