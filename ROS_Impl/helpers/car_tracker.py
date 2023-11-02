import cv2
import numpy as np
import torch

# import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from helpers.sort import *

def filter_cars(outputs, img):
    cls = outputs['instances'].pred_classes
    scores = outputs["instances"].scores
    masks = outputs['instances'].pred_masks
    box = outputs['instances'].pred_boxes.to(torch.device("cpu")).tensor.numpy()
    
    # non car elements
    idx_to_remove = (cls != 2).nonzero(as_tuple=False).flatten().tolist()

    # delete corresponding arrays
    cls = np.delete(cls.cpu().numpy(), idx_to_remove)
    scores = np.delete(scores.cpu().numpy(), idx_to_remove)
    masks = np.delete(masks.cpu().numpy(), idx_to_remove, axis=0)
    box = np.delete(box, idx_to_remove, axis=0)

    # convert back to tensor and move to cuda
    cls = torch.tensor(cls).to('cuda:0')
    scores = torch.tensor(scores).to('cuda:0')
    masks = torch.tensor(masks).to('cuda:0')
    box = torch.tensor(box).to('cuda:0')

    # create new instance obj and set its fields
    obj = detectron2.structures.Instances(image_size=(img.shape[0], img.shape[1]))
    box_obj = detectron2.structures.Boxes(box)
    obj.set('pred_classes', cls)
    obj.set('scores', scores)
    obj.set('pred_masks', masks)
    obj.set('pred_boxes', box_obj)

    return obj


def track(img):
    outputs = filter_cars(predictor(img),img)
    detection = torch.cat((outputs.pred_boxes.tensor,outputs.scores.unsqueeze(dim=1)),1).cpu().numpy() # Boxes and scores
    track_bbs_ids = tracker.update(detection)  # update SORT
    #Visualize
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    for box in track_bbs_ids:
        v.draw_box(box[:-1])
        v.draw_text(str(box[-1]),tuple(box[:2]))
    v = v.get_output()
    out = v.get_image()[:, :, ::-1]
    cv2.imshow("frame1",out)
    cv2.waitKey(0)

if __name__ == 'main':
    tracker = Sort()

    frame1 = cv2.imread("test.png")
    frame2 = cv2.imread("test2.png")
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    track(frame1)
    track(frame2)


