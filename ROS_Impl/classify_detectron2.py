#!/usr/bin/env python3
import math
import time

import rospy
from sensor_msgs.msg import CompressedImage, Image

from PIL import Image as IMG

# OpenCV
import cv2
from cv_bridge import CvBridge, CvBridgeError

import time

import os
import math
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import torch, torchvision

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import rospkg

from fub_cnn_rosnode_mrcnn.msg import MrcnnResult
from fub_cnn_rosnode_mrcnn.msg import MrcnnResultList

# tracking
from helpers.car_tracker import filter_cars
from helpers.sort import *
from helpers.car_frame_dict import carFrames_dict

class Classifier():

    def __init__( self):
        self.saveImages = False
        self.isReady = False
        self.isRunning = False
        self.isDone = True
        self.image_sk = None
        self.cvBridge = CvBridge()
        # Root directory of the project
        rospack = rospkg.RosPack()
        self.ROOT_DIR = rospack.get_path('fub_cnn_rosnode_mrcnn')
        self.tt0 = time.time()


    def run(self):    
        # global pubVizResults, classNames, model, isReady, isRunning, ROOT_DIR, IMAGE_DIR

        rospy.init_node('mrcnn_classifier', anonymous=True)
        rate = rospy.Rate(30) # hz
        ID = str(abs(hash(rospy.get_caller_id())) % (10 ** 8))
        rospy.loginfo('mrcnn_classifier ' + str(ID) + ': initialized')

        #rospy.Subscriber('/sensors/broadrreachcam_front/image_compressed/compressed', CompressedImage, self.callbackCompressedImage)
        rospy.Subscriber('/sensors/hella/image', Image, self.callbackImage)
        # rospy.Subscriber('/sensors/pg_left/image_raw/compressed', CompressedImage, self.callbackCompressedImage)
        #rospy.Subscriber('/sensors/broadrreachcam_front/image_rect', Image, self.callbackImage)
        self.pubVizResults = rospy.Publisher("/mrcnn/result_image", Image, queue_size=1)
        self.pubResults = rospy.Publisher("/mrcnn/results", MrcnnResultList, queue_size=1)

        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(cfg)
        tracker = Sort() # Tracker

        print ("MRCNN loaded")
        self.isReady = True

    # def convert_cv2_to_ros_msg(self, cv2_data, image_encoding='bgr8'):
    #     """
    #     Convert from a cv2 image to a ROS Image message.
    #     """
    #     return self._cv_bridge.cv2_to_imgmsg(cv2_data, image_encoding)

        # Dictionary for Frames
        CarFrames = carFrames_dict()

        while not rospy.is_shutdown():

            if self.isRunning:
                t0 = time.time()
                predictions = self.predictor(self.image_sk)
                instances = filter_cars(predictions,self.image_sk) # get Cars
                detection = torch.cat((instances.pred_boxes.tensor,instances.scores.unsqueeze(dim=1)),1).cpu().numpy() # Boxes and scores
                track_bbs_ids = tracker.update(detection)  # update SORT
                predictions = CarFrames.update(track_bbs_ids,self.image_sk) # update Dictionary

                metaDataCatalog = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

                # Visualie Results
                if self.pubVizResults.get_num_connections() > 0:
                    v = Visualizer(self.image_sk[:, :, ::-1], metaDataCatalog, scale=1.2)
                    for box in track_bbs_ids:
                        car_id = box[-1]
                        try:
                            pred = predictions[car_id]
                        except:
                            pred = "null"
                        label = [car_id,pred]
                        v.draw_box(box[:-1])
                        v.draw_text(str(label),tuple(box[:2]))
                    v = v.get_output()
                    out = v.get_image()[:, :, ::-1]
                    print ("has figure")
                    img = Image()
                    #image = IMG.fromarray(out)
                    image = IMG.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB).astype(np.uint8))
                    img.width , img.height = image.size
                    img.data = image.tobytes()
                    img.encoding = "rgb8"
                    self.pubVizResults.publish(img)
                    if self.saveImages:
                        pass

                if self.pubResults.get_num_connections() > 0:
                    classNames = metaDataCatalog.get("thing_classes", None)
                    boxes = instances.pred_boxes.tensor.numpy()
                    scores = instances.scores.numpy()
                    classes = instances.pred_classes.numpy()
                    # masks = instances.pred_masks.numpy()

                    resultList = MrcnnResultList()
                    for i in range(len(scores)):
                        result = MrcnnResult()
                        result.roi = boxes[i]
                        #result.mask = [item for sublist in masks[i] for item in sublist]
                        result.classId = classes[i]
                        result.className = classNames[i]
                        result.score = scores[i]
                        resultList.results.append(result)
                    self.pubResults.publish(resultList)
                    pass

                t1 = time.time()
                print(t1 - t0)

                self.isRunning = False
                self.isDone = True
                print ("processing image done")

    def callbackCompressedImage(self, img):
        if not self.isReady or (self.isRunning and not self.isDone):
            return
        self.isDone = False
        #### direct conversion to CV2 ####
        np_arr = np.fromstring(img.data, np.uint8)
        image_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        self.image_sk = skimage.img_as_ubyte(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        self.isRunning = True


    def callbackImage(self, img):
        if not self.isReady or (self.isRunning and not self.isDone):
            return
        self.isDone = False
        if img.encoding == '8UC3':
            img.encoding = 'rgb8'
        image_cv = self.cvBridge.imgmsg_to_cv2(img, "bgr8")
        self.image_sk = skimage.img_as_ubyte(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

        
        # cv2.imshow("name", self.image_sk)
        # cv2.waitKey(1)

        self.isRunning = True


    # def renderScene(timerStats):    
    #     global seq

    #     seq += 1
    #     currentTime = rospy.get_rostime()

def main():
  mrcnnClassifier = Classifier()
  mrcnnClassifier.run()

if __name__ == '__main__':
  main()
