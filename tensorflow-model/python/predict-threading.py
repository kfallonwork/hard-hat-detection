# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
"""Sample prediction script for TensorFlow 2.x."""
import sys
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection import ObjectDetection
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from object_detection import ObjectDetection
from io import BytesIO
import cv2
from threading import Thread

MODEL_FILENAME = '../../tensorflow-model/model.pb'
LABELS_FILENAME = '../../tensorflow-model/labels.txt'


class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow"""

    def __init__(self, graph_def, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            input_data = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(graph_def, input_map={"Placeholder:0": input_data}, name="")

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=float)[:, :, (2, 1, 0)]  # RGB -> BGR

        with tf.compat.v1.Session(graph=self.graph) as sess:
            output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
            outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis, ...]})
            return outputs[0]
        
    def draw_results(self, results, image):
        h, w, ch = np.array(image).shape
        print("entering prediction loop")
        converted_list = []
        for prediction in results:
            if (prediction["probability"]*100) > 50:
                left = prediction["boundingBox"]["left"] * w
                top = prediction["boundingBox"]["top"]  * h
                height = prediction["boundingBox"]["height"] *h
                width =  prediction["boundingBox"]["width"]  *w
                print("height: " + str(height) + ", width: " + str(width))
                start = (int(left), int(top))
                end = (int(left+width),int(top+height))
                text_pos = (int(left) - 2, int(top) - 2)
                string = prediction["tagName"] + ": " + str(round(prediction["probability"]*100,2))
                converted_list.append((start, end, string, text_pos))
        return converted_list
    
class PerformOD:

    # def __init__(self, exchange: VideoStream, language=None):
    def __init__(self):
        self.graph_def = tf.compat.v1.GraphDef()
        self.annotations = []
        with tf.io.gfile.GFile(MODEL_FILENAME, 'rb') as f:
            self.graph_def.ParseFromString(f.read())

    # Load labels
        with open(LABELS_FILENAME, 'r') as f:
            labels = [label.strip() for label in f.readlines()]
        self.od_model = TFObjectDetection(self.graph_def, labels)
        self.exchange = None
        self.stopped = False
        

    def start(self):
        Thread(target=self.od, args=()).start()
        return self

    def set_exchange(self, video_stream):
        self.exchange = video_stream

    def od(self):
       
        while not self.stopped:
            if self.exchange is not None:
                frame = self.exchange.frame
            
                frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame2, 'RGB')
                results = self.od_model.predict_image(img)
                annotated_image = self.od_model.draw_results(results, frame)
                self.annotations = annotated_image
                # for answer in annotated_image:
                #     cv2.rectangle(frame, answer[0], answer[1], (0, 0, 0), 1)
                #     cv2.putText(frame,answer[2], (answer[3]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)

    # def ODoneframe(self):
    #     if self.exchange is not None:
    #         frame = self.exchange.frame
    #         frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         img = Image.fromarray(frame2, 'RGB')
    #         results = self.od_model.predict_image(img)
    #         annotated_image = self.od_model.draw_results(results, frame)
    #         for answer in annotated_image:
    #             print(answer)
    #             cv2.rectangle(frame, answer[0], answer[1], (0, 0, 0), 1)
    #             cv2.putText(frame,answer[2], (answer[3]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)

class VideoStream: 
    """Class for CV2 video capture. The start() method will create a new 
thread to read the video stream"""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        #self._boxes = None
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def get_video_dimensions(self):
        width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return int(width), int(height)



def main():
    exchange = VideoStream(0).start()
    OD = PerformOD().start()
    OD.set_exchange(exchange)
    #OD = PerformOD()
    while True:  # Begins a loop for the real-time OCR display
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            break

        frame = exchange.frame 
        
        for answer in OD.annotations:
            cv2.rectangle(frame, answer[0], answer[1], (0, 0, 0), 1)
            cv2.putText(frame,answer[2], (answer[3]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
        
        cv2.imshow("Video Get Frame", frame)

if __name__ == '__main__':
        main()
