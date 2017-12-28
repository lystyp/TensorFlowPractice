import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.graph_util import convert_variables_to_constants


SAVE_FLAG = 1 # if 0 restore, else if 1 save, else if 2 error version

data = np.arange(start = 1, stop = 2, step = 1,  dtype=np.int32) # 1
data2 = np.arange(start = 2, stop = 3, step = 1,  dtype=np.int32) # 2


if SAVE_FLAG == 1 :
    # 先定義好graph
    x = tf.placeholder(tf.int32, [1], name="x")
    c = tf.constant(5,dtype=tf.int32, name = "c");
    y = tf.add(x, c, name="y")  # y = x + 5


    v1 = tf.Variable(data, name="v1")
    z = tf.add(v1, y, name="z")   

    # graph都建好之後之後定義初始化graph的操作
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)   # 跑初始化
        graph_def = sess.graph.as_graph_def()   # 取得 graph_def
        graph = convert_variables_to_constants(sess, graph_def, ["z"])  # 從 "z" 節點開始找相關聯的節點，把 graph_def 裡面所有與z相關的節點是 Variable 的話通通轉成 constant 固定死，
                                                                        # 並轉成新圖 graph
        tf.train.write_graph(graph, '.', 'graph.pb', as_text=False) # 存graph

elif SAVE_FLAG == 2 : # 會error的版本，變數轉常數的graph不能包含assign
    x = tf.placeholder(tf.int32, [1], name="x")
    c = tf.constant(5,dtype=tf.int32, name = "c");
    y = tf.add(x, c, name="y")  # y = x + 5


    v1 = tf.Variable(data, name="v1")
    z = tf.add(v1, y, name="z")   

    v2 = tf.Variable(data2, name="v2")
    assign = tf.assign(v2,z, name = "assign")   # 這裡assign的input有兩個，第一個input必須是可以改變的值，也就是Variable(他的dtype = int32_ref，一般的tensor的dtype = int32)
                                                # 把所有節點轉常數，存圖都不會有問題，問題會發生在讀圖的時候，讀圖的時候把所有節點讀回來，assign也被當作一個執行操作的tensor讀回來，
                                                # 這時候問題來了，assign節點讀進來後，要讀跟他相連的input節點 "v2" 跟 "z"，還記得assign的第一個input一定要是 dtype = int32_ref 吧?
                                                # 但是 "v2" 已經被轉換成constant了啊(dtype = int32)，所以會導致格式不合的錯誤
                                                # 想想也有道理啊，我都已經把圖鎖死了，怎麼可能還可以讓你用assign的操作來改變已經鎖死的節點的值
                                                # 附上錯誤訊息
                                                # ValueError: graph_def is invalid at node 'assign': Input tensor 'v2:0' Cannot co
                                                # nvert a tensor of type int32 to an input of type int32_ref.

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        graph_def = sess.graph.as_graph_def()
        graph = convert_variables_to_constants(sess, graph_def, ["assign"])
        tf.train.write_graph(graph, '.', 'graph.pb', as_text=False)


elif SAVE_FLAG == 0 :
    with tf.Session() as sess:
        with open('./graph.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read()) 
            tf.import_graph_def(graph_def, name='')  


        x = sess.graph.get_tensor_by_name("x:0")
        v1 = sess.graph.get_tensor_by_name("v1:0")
        z = sess.graph.get_tensor_by_name("z:0")
        print(x)
        print(v1)
        print(z)
        print(sess.run(z,  {x:data2}))

