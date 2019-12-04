# Source: https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html

import cv2


def showKeyPoints(imageName: str):
    image = cv2.imread(imageName)
    grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    keyPoints = sift.detect(grayscaleImage, None)

    print(keyPoints)
    # imageWithKeyPoints = cv2.drawKeypoints(grayscaleImage, keyPoints, image)    # Prints the grayscale image.
    imageWithKeyPoints = cv2.drawKeypoints(image, keyPoints, image)    # Prints the colored image.

    cv2.imshow("Image with key points", imageWithKeyPoints)
    cv2.waitKey(0)


imageNames = ["images/goldengate/goldengate-00.png", "images/goldengate/goldengate-01.png", "images/goldengate/goldengate-02.png", "images/goldengate/goldengate-03.png", "images/goldengate/goldengate-04.png", "images/goldengate/goldengate-05.png"]
for anImageName in imageNames:
    showKeyPoints(anImageName)
