# Preparing a TF model for usage in Android
# By Omid Alemi - Jan 2017
# Works with TF r1.0

import sys
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


MODEL_NAME = 'tfdroid'

# Freeze the graph

input_graph_path = MODEL_NAME+'.pbtxt'
checkpoint_path = './'+MODEL_NAME+'.ckpt'
input_saver_def_path = ""
input_binary = False
output_node_names = "O"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = 'frozen_'+MODEL_NAME+'.pb'
output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'
clear_devices = True

# http://blog.csdn.net/czq7511/article/details/72452985
freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path,
                          output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")



# Optimize for inference

input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["I"], # an array of the input node(s)
        ["O"], # an array of output nodes
        tf.float32.as_datatype_enum)

# Save the optimized graph

f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
f.write(output_graph_def.SerializeToString())

# tf.train.write_graph(output_graph_def, './', output_optimized_graph_name)                    