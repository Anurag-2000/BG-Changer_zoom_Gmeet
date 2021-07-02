import cv2 as cv
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os


frameWidth= 640
frameHight=480

cap = cv.VideoCapture(0)

cap.set(3,frameWidth)
cap.set(3,frameHight)
# cap.set(10,150)
segmentor = SelfiSegmentation()
fpsreader = cvzone.FPS()
imgbg = cv.imread('F:\pycodes\\background_remove\\bg\\1.jpg')
list_of_bgs = os.listdir('F:\pycodes\\background_remove\\bg')

print(list_of_bgs)
listimgs = []
for bgs_path in list_of_bgs:
    img = cv.imread(f'F:\pycodes\\background_remove\\bg\\{bgs_path}')
    listimgs.append(img)
lenghtmax = len(listimgs)
index = 0

while True:
    isTrue, frame = cap.read()
    frame = cv.resize(frame, (frameWidth,frameHight))
    frameout = segmentor.removeBG(frame,listimgs[index],threshold=0.7)
    imgStacked = cvzone.stackImages([frame,frameout],2,1)
    _,imgStacked = fpsreader.update(imgStacked)


    cv.imshow('video f by f', imgStacked)
    key = cv.waitKey(1)
    if key == ord('a'):
        index = (index -1)%lenghtmax
    elif key == ord('d'):
        index = (index +1)%lenghtmax
    elif key == ord('q'):
        break