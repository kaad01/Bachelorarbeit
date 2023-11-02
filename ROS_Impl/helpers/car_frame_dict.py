from helpers.Backlight_Recognition import load_model, Transformer
import torch
import cv2
from PIL import Image
import numpy as np

class carFrames_dict:
    def __init__(self):
        self.carFrames = {} # Dicitionary with alĺ Frames linked to there IDs
        self.model = load_model("/mnt/HDD/Dokumente/Bachelorarbeit/modelle/saved_models/newmodel_noNorm").cuda() # give path to Modell
        self.Transformer = Transformer()
        self.classes = {
                0:"OOO",
                1:"BOO",
                2:"OLO",
                3:"BLO",
                4:"OOR",
                5:"BOR",
                6:"OLR",
                7:"BLR"
            }

    def update(self,tracker,im):
        check_cars = [] # car ids of the cars that have 8 Frames; those cars are getting checked this time
        car_ids = []
        """Get Image"""
        for x1,y1,x2,y2,car_id in tracker:
            x1,x2,y1,y2 = int(x1), int(x2), int(y1), int(y2)
            crop_img = im[y1:y2, x1:x2]

            # append new Frame
            if car_id in self.carFrames:
                self.carFrames[car_id].append(crop_img)
            else:
                self.carFrames[car_id] = [crop_img]

            # check if a new car has 8 frames or more
            if len(self.carFrames[car_id]) >= 8:
                check_cars.append(car_id)

            car_ids.append(car_id)

        # delete every Car in Dict that is not in this Frame
        for id in list(self.carFrames.keys()):
            if id not in car_ids:
                print(id)
                del self.carFrames[id]

        """send data to model"""
        predictions = {}
        for car_id in check_cars:
            frames = self.carFrames[car_id]
            # predict state of car
            frames = self.Transformer.transform(frames)

            del self.carFrames[car_id][0] # delete first Frame, because we do not need more than 8 Frames per Car
            #frames = frames[0::2] we dont need this
            prediction = self.model(frames)
            prediction = self.classes[prediction]
            predictions[car_id] = prediction
        
        
        return predictions
