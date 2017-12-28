import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

SAVE_FLAG = 0 # if 0 restore,else save

data = np.arange(start = 3, stop = 4, step = 1,  dtype=np.int32)
data2 = np.arange(start = 5, stop = 6, step = 1,  dtype=np.int32)

# 做一個方程式 y = x + 5

if SAVE_FLAG == 1 :
    with tf.Session() as sess:
    
        x = tf.placeholder(tf.int32, [1], name="x")
        c = tf.constant(5,dtype=tf.int32, name = "c");
        y = tf.add(x, c, name="y") #  data depends on the input data
        saved_result= tf.Variable(data, name="saved_result")
        do_save=tf.assign(saved_result,y, name = "assign")

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        print("各變數的型態")
        print(x)
        print(c)
        print(y)
        print(saved_result)
        print(do_save)
        # 存graph
        print("Save graph path >>> " )
        print( tf.train.write_graph(sess.graph_def, "tmp", "test.pb", False)) 

        # 算結果
        result,_=sess.run([y,do_save], {x: data}) # calculate output1 and assign to 'saved_result'
        print( "resule >>  ")
        print(result)

        # 存變數
        saver = tf.train.Saver()
        saver.save(sess,"./tmp/checkpoint.data")

else :
    print(" >>>>>>>>>>>>>>>>>>>>> ")
    with tf.Session() as sess2:
        # new_saved_result = tf.Variable(tf.constant(0,dtype=tf.int32), name="saved_result") # 為什麼要宣告這個變數呢?因為ckpt只能把值存回變數，不能存回tensor

        # init_op = tf.global_variables_initializer() # 初始化依定要在變數定義完才定義，不然初始化晚定義的變數都不會初始化了
        # sess2.run(init_op)

        print("load graph")
        with tf.gfile.FastGFile("tmp/test.pb",'rb') as f:  # 讀pb檔 !!! 'rb'應該是指讀binary檔，記得之前存graph要存binary這邊才不會error
            graph_def = tf.GraphDef()                   # 建預設的graph
            graph_def.ParseFromString(f.read())         # 用預設的graph讀檔
            sess2.graph.as_default()                     # 在這個context底下的graph把他給設成預設(初始化的意思?)
            tf.import_graph_def(graph_def, name='')     # 把剛剛讀的graph塞到目前的context底下
        
        # saved_result= tf.Variable(data, name="saved_result") # 為何不能放這裡? 因為讀完graph之後就有一個叫saved_result的tensor了，所以放這裡的話會被取成saved_result1

        persisted_x = sess2.graph.get_tensor_by_name("x:0")
        persisted_c = sess2.graph.get_tensor_by_name("c:0")
        persisted_y = sess2.graph.get_tensor_by_name("y:0")
        # 如果有宣告new_saved_result，這個就有點像get by reference一樣，persisted_result跟new_saved_result抓到的是同一個參考，我改其中一個，另一個會跟著變
        persisted_result = sess2.graph.get_tensor_by_name("saved_result:0")      # 用string名稱來讀graph裡的節點，but為什麼要加一個 :0
        persisted_do_save = sess2.graph.get_tensor_by_name("assign:0")

        # # 初始化
        # init_op = tf.global_variables_initializer() # 初始化依定要在變數定義完才定義，不然初始化晚定義的變數都不會初始化了
        # sess2.run(init_op)

        # saver = tf.train.Saver() # 'Saver' misnomer! Better: Persister!
        # print("load data")
        # saver.restore(sess2, "./tmp/checkpoint.data")  # now OK
        # print(sess2.run(persisted_result))
        # print("DONE")

        # 看來好像確定可以把ckpt塞回變數，那原本是常數跟placeholder的呢? > 不行，因為ckpt只有存變數，感覺上是要自己另外塞值
        # 測試
        # 1.改 變數的值，對應的tensor會變嗎? > 會
        # 2.改tensor的值變數會跟著變嗎? > 會
        # 3.變數跟tensor會自動配對嗎?用變數跑graph運算試試看? > 變數跟變數對應的tensor值都不會變，都維持是ckpt讀進來的值


        # 從下面這裡會發現，雖然從graph讀進來後都是tensor，但還是保有placeHolder和constnat的屬性
        print(sess2.run(persisted_x, {persisted_x:data2})) # placeholder一樣要feed一個值給他才可以跑
        print(sess2.run(persisted_c)) # 常數直接就有值了
        print(sess2.run(persisted_y, {persisted_x:data2}))  # 這裡因為有讀graph進來，知道y = x + c
        # print(sess2.run(persisted_result)) 一個沒有給初始值的tensor，會error
        print(sess2.run(persisted_do_save, {persisted_x:data2})) # 這個也可以正常跑

        # 重點在這裡
        # 有2種case
        # 1.如果要從ckpt讀參數，
        #    > 就必須宣告new_saved_result，因為saver要找graph中有變數才可以執行restore，但這時候new_saved_result會是宣告成一個新節點
        #   後面用persisted_result = sess2.graph.get_tensor_by_name("saved_result:0") 來節點就不是取到上次存的graph的saved_result了，所以跑graph怎麼樣都無法改變這個值，都維持ckpt讀的值
        #    > 如果不宣告new_saved_result，sess2.graph.get_tensor_by_name("saved_result:0")讀進來就只是個有變數性質的tensor，saver找不到變數可以restore，會error
        #         P.s. saver裡面如果有多個變數值可以restore，如果restore的是全部變數，graph只要有一個變數有對到就可以restore了，但graph裡面不能有saver裡面沒有的變數，不然restore時會error，
        #         也就是，saver丟進來可以找不到變數塞沒關係，但graph不能有空的變數沒有被saver塞
        #   所以1的結論是，讀graph進來後的變數，沒辦法用saver來把變數塞回去的樣子
        #
        # 2.不從ckpt讀變數，所以我只是單純要他的graph，那就沒必要宣告new_saved_result，就直接讀進來就可以用啦，雖然他有變數的性質，有嗎?感覺上只是一班的tensor
        #   這裡就可以把節點讀進來便tensor之後就直接用啦

        # print(sess2.run(new_saved_result))
        print(sess2.run(persisted_result))

    

# Question !!!
# 不同的session到底怎麼分別?為何在上面session的變數在這裡可以讀，是因為同grpah的關係?
# 我發現在with as 底下宣告的變數，一樣是全域的，是因為同一個graph嗎

# 要怎麼讀graph之後把變數也塞回去呢? 
# 兩種方法 > http://www.jianshu.com/p/091415b114e2
# > FREEZE_GRAPH
# > CONVERT_VARIABLES_TO_CONSTANTS






# 下面這些是網路上的sample code，上面是我自己改的
# https://www.bountysource.com/issues/29393251-how-to-use-tf-train-write_graph-and-tf-import_graph_def-it-seems-that-it-does-not-work

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import tensorflow as tf
# import numpy as np
# from tensorflow.python.platform import gfile

# data = np.arange(10,dtype=np.int32)
# with tf.Session() as sess:
#   print("# build graph and run")
#   input1= tf.placeholder(tf.int32, [10], name="input")
#   output1= tf.add(input1, tf.constant(100,dtype=tf.int32), name="output") #  data depends on the input data
#   saved_result= tf.Variable(data, name="saved_result")
#   do_save=tf.assign(saved_result,output1)
#   tf.global_variables_initializer()
# #   os.system("rm -rf /tmp/load")
#   tf.train.write_graph(sess.graph_def, "tmp", "test.pb", False) #proto
#   # now set the data:
#   result,_=sess.run([output1,do_save], {input1: data}) # calculate output1 and assign to 'saved_result'
#   saver = tf.train.Saver(tf.global_variables())
#   saver.save(sess,"./tmp/checkpoint.data")

# with tf.Session() as persisted_sess:
#   print("load graph")
#   with gfile.FastGFile("tmp/test.pb",'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     persisted_sess.graph.as_default()
#     tf.import_graph_def(graph_def, name='')
#   print("map variables")
#   persisted_result = persisted_sess.graph.get_tensor_by_name("saved_result:0")
# #   tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES,persisted_result)      # 關於add_to_collection http://blog.csdn.net/u012436149/article/details/53894354
#   try:
#     saver = tf.train.Saver(tf.global_variables()) # 'Saver' misnomer! Better: Persister!
#   except:pass
#   print(tf.global_variables())
#   print("load data")
#   saver.restore(persisted_sess, "./tmp/checkpoint.data")  # now OK
#   print(persisted_result.eval())
#   print("DONE")


