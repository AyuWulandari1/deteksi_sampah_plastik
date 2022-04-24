import tensorflow as tf
import numpy as np
from core_service import visualization_utils as vis_util
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

def set_model(model_name):
	model_name = model_name
	path_to_ckpt = 'core_service/'+model_name + '/frozen_inference_graph.pb'
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.compat.v1.GraphDef()
		with tf.io.gfile.GFile(path_to_ckpt, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	category_index = {
    1: {'id': 1, 'name': 'Botol Plastik'},
    2: {'id': 2, 'name': 'Gelas Plastik'},
    3: {'id': 3, 'name': 'Sendok'},
    4: {'id': 4, 'name': 'Styrofoam'},
}
	return detection_graph, category_index

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)