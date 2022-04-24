import cv2 
import numpy as np 
from core_service import visualization_utils as vis_util
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Stream():
    def __init__(self, camera_src):
        self.camera_src = camera_src
        self.camera = None

    def gen_frames(self):
        
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                while True:
                    if self.camera is not None :
                        ret, image_np = self.camera.read()

                        image_np_expanded = np.expand_dims(image_np, axis=0)
                        image_tensor =self. detection_graph.get_tensor_by_name('image_tensor:0')
                        boxes =self. detection_graph.get_tensor_by_name('detection_boxes:0')
                        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                        (boxes, scores, classes, num_detections) = sess.run(
                            [boxes, scores, classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            self.category_index,
                            use_normalized_coordinates=True,
                            min_score_thresh=0.7,
                            line_thickness=8)

                        if not ret:
                            break
                        
                        ret, buffer = cv2.imencode('.jpg', image_np)
                        image_np = buffer.tobytes()
                        yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + image_np + b'\r\n')

    def close(self):
        if self.camera is not None :
            self.camera.release()
            self.camera = None

    def open(self):
        PATH_TO_CKPT = 'core_service/Inception/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.category_index = {
                1: {'id': 1, 'name': 'Botol Plastik'},
                2: {'id': 2, 'name': 'Gelas Plastik'},
                3: {'id': 3, 'name': 'Sendok'},
                4: {'id': 4, 'name': 'Styrofoam'},
            }
        self.camera = cv2.VideoCapture(self.camera_src)

    def status(self):
        return self.camera is not None


