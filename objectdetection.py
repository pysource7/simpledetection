import cv2
import numpy as np
import urllib.request
import os

class ObjectDetection:
    def __init__(self):
        dnn_folder = "/content/dnn"
        
        print("Downloading DNN model")
        if not os.path.exists(dnn_folder):
          os.makedirs(dnn_folder)
        urllib.request.urlretrieve("https://pjreddie.com/media/files/yolov3-spp.weights", "/content/dnn/yolov3-spp.weights")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-spp.cfg", "/content/dnn/yolov3-spp.cfg")
        
        net = cv2.dnn.readNet("/content/dnn/yolov3-spp.weights", "/content/dnn/yolov3-spp.cfg")
        # Enable GPU CUDA
        # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(608, 608), scale=1/255)

        self.classes = []
        with open("dnn_model/classes.txt", "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
    def detect(self, img):
        class_ids, scores, boxes = self.model.detect(img, nmsThreshold=0.4)
        return class_ids, scores, boxes
