# Source: https://docs.opencv.org/trunk/dc/dc3/tutorial_py_matcher.html
# Feature matching with ORB (Oriented FAST and rotated BRIEF) and BFMatcher.

import numpy as np
import cv2
import matplotlib.pyplot as plt


def matchFeaturesWithORB(imageName1, imageName2):
    image1 = cv2.imread(imageName1)
    image2 = cv2.imread(imageName2)

    # MARK: Extract features with ORB.
    # Initialize the ORB detector.
    orb = cv2.ORB_create()

    # Find key points and descriptors.
    keyPoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keyPoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # MARK: Match features
    # Creates a BFMatcher object.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches in the order of distance.
    sortedMatches = sorted(matches, key=lambda x:x.distance)

    # Draw the first 10 matches.
    imageWithSortedMatches = cv2.drawMatches(image1, keyPoints1, image2, keyPoints2, sortedMatches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Plot with either matplotlib or opencv. matplotlib is laggy and is not recommended.
    cv2.imshow("Sorted matches", imageWithSortedMatches)
    cv2.waitKey(0)
    # plt.imshow(imageWithSortedMatches)
    # plt.show()


matchFeaturesWithORB("images/goldengate/goldengate-00.png", "images/goldengate/goldengate-01.png")
