from YoloV7ObjectDetection import *

weightsFile = "pretrained_models/Yolov7/Yolov7-544_v1.pt"
testImage = "test/val_1.jpg"
imageSize = 544

confScore = 0.5
maxNumBoxes = 10
overlap = 0.45

detector = YoloV7Detector()

# detector.detect(weightsFile, testImage, imageSize, confScore, maxNumBoxes, overlap)
detector.screenCapture(weightsFile, imageSize, confScore, overlap)
# detector.windowCapture()
