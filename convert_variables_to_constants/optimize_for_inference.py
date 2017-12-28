

import sys
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

# 測試convert_variables_to_constants是否還需要最佳化
# 測試結果檔案內容會變欸

with open('./graph.pb', 'rb') as f:
    input_graph_def = tf.GraphDef()
    input_graph_def.ParseFromString(f.read()) 

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["x"], # an array of the input node(s)
        ["z"], # an array of output nodes
        tf.int32.as_datatype_enum)

# Save the optimized graph

f = tf.gfile.FastGFile("optimized_graph.pb", "w")
f.write(output_graph_def.SerializeToString())
