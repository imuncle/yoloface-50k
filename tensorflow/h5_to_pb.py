import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def h5_to_pb(h5_save_path):
    model = tf.keras.models.load_model(h5_save_path, compile=False)
    model.summary()
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./pb",
                      name="model.pb",
                      as_text=False)   #可设置.pb存储路径


h5_to_pb('./yoloface.h5')   #此处填入.h5路径


# import tensorflow.compat.v1 as tf1
 
# tf1.reset_default_graph()
# tf1.keras.backend.set_learning_phase(0)  # 调用模型前一定要执行该命令
# tf1.disable_v2_behavior()  # 禁止tensorflow2.0的行为

# hdf5_pb_model = tf1.keras.models.load_model('yoloface.h5')
# def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
#     graph = session.graph
#     with graph.as_default():
#         #         freeze_var_names = list(set(v.op.name for v in tf1.global_variables()).difference(keep_var_names or []))
#         output_names = output_names or []
#         #         output_names += [v.op.name for v in tf1.global_variables()]
#         print("output_names", output_names)
#         input_graph_def = graph.as_graph_def()
#         #         for node in input_graph_def.node:
#         #             print('node:', node.name)
#         print("len node1", len(input_graph_def.node))
#         if clear_devices:
#             for node in input_graph_def.node:
#                 node.device = ""
#         frozen_graph = tf1.graph_util.convert_variables_to_constants(session, input_graph_def,
#                                                                      output_names)
 
#         outgraph = tf1.graph_util.remove_training_nodes(frozen_graph)  # 云掉与推理无关的内容
#         print("##################################################################")
#         for node in outgraph.node:
#             print('node:', node.name)
#         print("len node1", len(outgraph.node))
#         return outgraph

# output_folder2 = './pb'
 
# frozen_graph = freeze_session(tf1.compat.v1.keras.backend.get_session(),
#                               output_names=[out.op.name for out in hdf5_pb_model.outputs])
# tf1.train.write_graph(frozen_graph, output_folder2, "model.pb", as_text=False)