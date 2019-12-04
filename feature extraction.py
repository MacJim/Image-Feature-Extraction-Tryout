import numpy as np
import cv2


def showKeyPoints(imageName: str):
    image = cv2.imread(imageName)
    grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    keyPoints = sift.detect(grayscaleImage, None)

    print(keyPoints)
    imageWithKeyPoints = cv2.drawKeypoints(grayscaleImage, keyPoints, image)

    cv2.imshow("Image with key points", imageWithKeyPoints)
    cv2.waitKey(0)


imageNames = ["goldengate/goldengate-00.png", "goldengate/goldengate-01.png", "goldengate/goldengate-02.png", "goldengate/goldengate-03.png", "goldengate/goldengate-04.png", "goldengate/goldengate-05.png"]
for anImageName in imageNames:
    showKeyPoints(anImageName)