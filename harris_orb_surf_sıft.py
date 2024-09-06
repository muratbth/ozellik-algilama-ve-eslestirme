import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread('manzara.png')
img2 = cv2.imread('beach.png')


gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


def harris_corner_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    return image


def sift_feature_matching(img1, img2):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches


def orb_feature_matching(img1, img2):
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches



# Harris Sonucu
harris_img1 = harris_corner_detection(img1.copy())
harris_img2 = harris_corner_detection(img2.copy())

# SIFT Sonucu
sift_result = sift_feature_matching(gray1, gray2)

# ORB Sonucu
orb_result = orb_feature_matching(gray1, gray2)



plt.figure(figsize=(15, 10))
plt.subplot(231), plt.imshow(cv2.cvtColor(harris_img1, cv2.COLOR_BGR2RGB)), plt.title('Harris Algılama - Görsel 1')
plt.subplot(232), plt.imshow(cv2.cvtColor(harris_img2, cv2.COLOR_BGR2RGB)), plt.title('Harris Algılama - Görsel 2')
plt.subplot(233), plt.imshow(sift_result), plt.title('SIFT Eşleşmeleri')
plt.subplot(234), plt.imshow(orb_result), plt.title('ORB Eşleşmeleri')


plt.tight_layout()
plt.show()
