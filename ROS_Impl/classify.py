#!/usr/bin/env python3
import math
import time

import rospy
from sensor_msgs.msg import CompressedImage, Image

# from visualization_msgs.msg import MarkerArray, Marker
# from std_msgs.msg import ColorRGBA, Header

# from sensor_msgs.msg import PointCloud2, PointCloud, NavSatFix, PointField
# import sensor_msgs.point_cloud2 as pc2
# import tf2_ros
# import tf
# import csv
# import os
# from os import listdir
# from os.path import isfile, join

# from thread import start_new_thread
# import re

# OpenCV
import cv2
from cv_bridge import CvBridge, CvBridgeError

import time

import os
import math
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

from mrcnn import coco
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

import rospkg

from fub_cnn_rosnode_mrcnn.msg import MrcnnResult
from fub_cnn_rosnode_mrcnn.msg import MrcnnResultList

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


        # """
        # Initialize the instance

        # :param camera_name: The camera name. One of (head_camera, right_hand_camera)
        # :param base_frame: The frame for the robot base
        # :param table_height: The table height with respect to base_frame
        # """
        # self.camera_name = camera_name
        # self.base_frame = base_frame
        # self.table_height = table_height

        # self.image_queue = Queue.Queue()
        # self.pinhole_camera_model = PinholeCameraModel()
        # self.tf_listener = tf.TransformListener()

        # camera_info_topic = "/io/internal_camera/{}/camera_info".format(camera_name)
        # camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)

        # self.pinhole_camera_model.fromCameraInfo(camera_info)

        # cameras = intera_interface.Cameras()
        # cameras.set_callback(camera_name, self.__show_image_callback, rectify_image=True)


    def run(self):    
        # global pubVizResults, classNames, model, isReady, isRunning, ROOT_DIR, IMAGE_DIR

        rospy.init_node('mrcnn_classifier', anonymous=True)
        rate = rospy.Rate(30) # hz
        ID = str(abs(hash(rospy.get_caller_id())) % (10 ** 8))
        rospy.loginfo('mrcnn_classifier ' + str(ID) + ': initialized')

        rospy.Subscriber('/sensors/broadrreachcam_front/image_compressed/compressed', CompressedImage, self.callbackCompressedImage)
        #rospy.Subscriber('/sensors/broadrreachcam_front/image_rect', Image, self.callbackImage)
        self.pubVizResults = rospy.Publisher("/mrcnn/result_image", Image, queue_size=1)
        self.pubResults = rospy.Publisher("/mrcnn/results", MrcnnResultList, queue_size=1)

        # TODO: check if needed
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(self.ROOT_DIR, "logs")

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(self.ROOT_DIR, "res/mask_rcnn_coco.h5")
        # # Download COCO trained weights from Releases if needed
        # if not os.path.exists(COCO_MODEL_PATH):
        #     utils.download_trained_weights(COCO_MODEL_PATH)

        # Directory of images to run detection on
        self.IMAGE_DIR = os.path.join(self.ROOT_DIR, "images_fu")

        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        # self.config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.classNames = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                    'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                    'teddy bear', 'hair drier', 'toothbrush']


        # # Load a random image from the images folder
        # file_names = next(os.walk(IMAGE_DIR))[2]
        # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

        # # Run detection
        # results = model.detect([image], verbose=1)

        # # Visualize results
        # r = results[0]
        # plot = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], classNames, r['scores'])
        # plot.savefig(os.path.join(ROOT_DIR, "src/fub_cnn_rosnode_mrcnn/test.png"))

        print ("MRCNN loaded")
        self.isReady = True

        while not rospy.is_shutdown():

            if self.isRunning:
                t0 = time.time()
                image = self.image_sk
                results = self.model.detect([image], verbose=0)
                r = results[0]
                fig = None
                if self.pubVizResults.get_num_connections() > 0:
                    plot, fig = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], self.classNames, r['scores'], show_mask=True, show_bbox=True)
                    print ("has figure")
                    canvas = fig.gca().figure.canvas
                    canvas.draw()
                    img = Image()
                    img.width, img.height = canvas.get_width_height()
                    img.data = canvas.tostring_rgb()
                    img.encoding = "rgb8"
                    self.pubVizResults.publish(img)
                    if self.saveImages:
                        plot.savefig(os.path.join(self.ROOT_DIR, "test.png"))
                        #plot.show()
                # else:
                #     print([ r['rois'], r['class_ids'], r['scores'] ])

                if self.pubResults.get_num_connections() > 0:
                    resultList = MrcnnResultList()
                    for i in range(len(r['class_ids'])):
                        result = MrcnnResult()
                        result.roi = r['rois'][i]
                        #result.mask = [item for sublist in r['masks'][i] for item in sublist]
                        result.classId = r['class_ids'][i]
                        result.className = self.classNames[result.classId]
                        result.score = r['scores'][i]
                        resultList.results.append(result)
                    self.pubResults.publish(resultList)

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
        self.image_sk = skimage.img_as_ubyte(cv2.cvtColor(image_cv[1:800, 128:1152], cv2.COLOR_BGR2RGB))
        self.isRunning = True


    def callbackImage(self, img):
        if not self.isReady or (self.isRunning and not self.isDone):
            return
        self.isDone = False

        image_cv = self.cvBridge.imgmsg_to_cv2(img, "bgr8")
        self.image_sk = skimage.img_as_ubyte(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

        # cv2.imshow("name", image_cv)
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
