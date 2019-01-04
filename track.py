
import sys
import copy
import cv2
import numpy as np
from keras.models import load_model

from utils.utils import Entity
from keras.models import model_from_json



def main():
    #file = "/media/rodrigo/Rodrigo/vtest.avi"
    file = "/media/rodrigo/Rodrigo/PAO/TownCentreXVID.avi"
    #file = "/media/rodrigo/Rodrigo/PAO/pedestrian_detection/aa.avi"
    iou=0.2
    track(file, iou)

def loadModelAndWeights(pathToModel,pathToWeights):
    json_file = open(pathToModel, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    #load model
    loaded_model = model_from_json(loaded_model_json)
    print("Loading model from " , pathToModel)
    
    # load weights into new model
    loaded_model.load_weights(pathToWeights)
    print("Loading weights from ", pathToWeights)
    print("Loaded model and weights from disk")
    model = loaded_model
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    
    return model
    
def overlap(box1, box2):
    """
    Check the overlap of two boxes
    """
    endx = max(box1[0] + box1[2], box2[0] + box2[2])
    startx = min(box1[0], box2[0])
    width = box1[2] + box2[2] - (endx - startx)

    endy = max(box1[1] + box1[3], box2[1] + box2[3])
    starty = min(box1[1], box2[1])
    height = box1[3] + box2[3] - (endy - starty)

    if (width <= 0 or height <= 0):
        return 0
    else:
        Area = width * height
        Area1 = box1[2] * box1[3]
        Area2 = box2[2] * box2[3]
        ratio = Area / (Area1 + Area2 - Area)

        return ratio


def track(video, iou):
    camera = cv2.VideoCapture(video)
    res, frame = camera.read()
    y_size = frame.shape[0]
    x_size = frame.shape[1]
    
    # Load CNN classification model
    #model = loadModelAndWeights("ped-18x36/modelcnn-18x36.json","ped-18x36/weights-18x36.h5")
    #model = loadModelAndWeights("kitti/modelVggV2-Kitti.json","kitti/savedweightsvgg-Kitti.h5")
    model = loadModelAndWeights("ped-64x64/modelcnn-64x64.json","ped-64x64/weights-64x64.h5")
    
    # Definition of MOG2 Background Subtraction
    bs = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    history = 20
    frames = 0
    counter = 0

    track_list = []
    cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
    while True:
        res, frame = camera.read()

        if not res:
            break
        # Train the MOG2 with first frames frame
        fg_mask = bs.apply(frame)

        if frames < history:
            frames += 1
            continue
        # Expansion and denoising the original frame
        th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check the bouding boxs
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if cv2.contourArea(c) > 3000:
                # Extract roi
                img = frame[y: y + h, x: x + w, :]
                #if using model 18 x36
                #rimg = cv2.resize(img, (36, 18), interpolation=cv2.INTER_CUBIC)
                #if using model 64x64
                rimg = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
                #if using model kitti
                #rimg = cv2.resize(img, (150, 150), interpolation=cv2.INTER_CUBIC)
                image_data = np.array(rimg, dtype='float32')
                image_data /= 255.
                roi = np.expand_dims(image_data, axis=0)
                flag = model.predict(roi)

                if flag[0][0] > 0.5:
                    e = Entity(counter, (x, y, w, h), frame)

                    # Exclude existing targets in the tracking list
                    if track_list:
                        count = 0
                        num = len(track_list)
                        for p in track_list:
                            if overlap((x, y, w, h), p.windows) < iou:
                                count += 1
                        if count == num:
                            track_list.append(e)
                    else:
                        track_list.append(e)
                    counter += 1

        # Check and update goals
        if track_list:
            tlist = copy.copy(track_list)
            for e in tlist:
                x, y = e.center
                if 10 < x < x_size - 10 and 10 < y < y_size - 10:
                    e.update(frame)
                else:
                    track_list.remove(e)
        frames += 1
        cv2.imshow("detection", frame)
        if cv2.waitKey(110) & 0xff == 27:
            break
    camera.release()


if __name__ == '__main__':
    main() 
