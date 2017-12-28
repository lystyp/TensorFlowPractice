import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import time
import numpy as np


initializer = tf.random_uniform_initializer(-2, 2)
v1 = tf.Variable(tf.constant(1.0,shape=[1,1]), name = "v1")
add = tf.add(v1,1)
update = tf.assign(v1,add)













path = "save/"

sv = tf.train.Supervisor(logdir = path)
with sv.managed_session() as session:
    for i in range(100):
        # if sv.should_stop():
        #     print("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")
        session.run(add)
        session.run(update)
        print(session.run(v1))







#note1 不必初始化
# http://cwiki.apachecn.org/pages/viewpage.action?pageId=10029499#Supervisor:长期训练帮手-一个简单方案