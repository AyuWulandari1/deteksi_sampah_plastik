from flask import Flask, render_template, Response, request, flash, redirect, url_for, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename
import numpy as np
import os,sys
import cv2
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sys.path.append("..")
from core_service import visualization_utils as vis_util
from core_service import backbone
from core_service.stream import Stream

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['SECRET_KEY'] = 'qwerty123z'

#
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

detection_graph, category_index = backbone.set_model('Inception')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/video_feed')
def video_feed():
    return Response(stream.gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

w, h = 480, 480
stream = Stream(camera_src=(0))

@app.route("/")
def index():
    camera = request.args.get("camera")
    if camera is not None and camera == 'off' and stream.status() == True:
        stream.close()
        flash("Camera turn off!", "info")
    elif camera is not None and camera == 'on' and stream.status() == False:
        stream.open()
        flash("Camera turn on!", "success")

    setting = dict(
        stream_on = stream.status(),
        w = w,
        h = h
    )
    return render_template("index.html", setting = setting)


#menu 2

@app.route("/menu2")
def menu2():
    return render_template("menu2.html")

@app.route('/uploads', methods=['POST'])
def uploads(): 
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                image_np = backbone.load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    min_score_thresh=0.7,
                    line_thickness=8)
                im = Image.fromarray(image_np)
                disk='static/uploads'
                filesToRemove = [os.path.join(disk,f) for f in os.listdir(disk)]
                for f in filesToRemove:
                    os.remove(f) 

                im.save('static/uploads/'+'I_'+filename)
    return render_template('menu2.html', filename='static/uploads/'+'I_'+filename)

@app.route('/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/'+'I_' + filename), code=301)

##menu2


#MENU 3
@app.route("/menu3")
def menu3():
    tes='TES'
    return render_template("menu3.html", tes=tes)

##MENU 3

if __name__ == '__main__':
    app.run( debug=True)