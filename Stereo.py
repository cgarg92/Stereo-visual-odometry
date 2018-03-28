import numpy as np
import cv2

ImT1_L = cv2.imread('/Volumes/Files/dataset/sequences/00/image_0/000000.png', 0)  # 0 flag returns a grayscale image
ImT1_R = cv2.imread('/Volumes/Files/dataset/sequences/00/image_1/000000.png', 0)

ImT2_L = cv2.imread('/Volumes/Files/dataset/sequences/00/image_0/000001.png', 0)
ImT2_R = cv2.imread('/Volumes/Files/dataset/sequences/00/image_1/000001.png', 0)

# cv2.imshow('ImT1_L', ImT1_L)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

block = 15

# emperical values from P1, P2 as suggested in Ocv documentation
P1 = 0  # block * block * 8
P2 = 0  # block * block * 32

disparityEngine = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=block, P1=P1, P2=P2)
ImT1_disparity = disparityEngine.compute(ImT1_L, ImT1_R).astype(np.float32)
cv2.imwrite('disparity.png', ImT1_disparity)
ImT1_disparityA = np.divide(ImT1_disparity, 16.0)

ImT2_disparity = disparityEngine.compute(ImT2_L, ImT2_R).astype(np.float32)
ImT2_disparityA = np.divide(ImT2_disparity, 16.0)
TILE_H = 10
TILE_W = 20
fastFeatureEngine = cv2.FastFeatureDetector_create()

#keypoints = fastFeatureEngine.detect(ImT1_L)
#ftDebug = ImT1_L
#ftDebug = cv2.drawKeypoints(ImT1_L, keypoints, ftDebug, color=(255,0,0))
#cv2.imwrite('ftDebug.png', ftDebug)

# 20x10 (wxh) tiles for extracting less features from images
H, W = ImT1_L.shape
kp = []
idx = 0
for y in range(0, H, TILE_H):
    for x in range(0, W, TILE_W):
        imPatch = ImT1_L[y:y + TILE_H, x:x + TILE_W]
        keypoints = fastFeatureEngine.detect(imPatch)
        for pt in keypoints:
            pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

        if (len(keypoints) > 10):
            keypoints = sorted(keypoints, key=lambda x: -x.response)
            for kpt in keypoints[0:10]:
                kp.append(kpt)
        else:
            for kpt in keypoints:
                kp.append(kpt)

ftDebug = ImT1_L
ftDebug = cv2.drawKeypoints(ImT1_L, kp, ftDebug, color=(255, 0, 0))
#cv2.imwrite('ftDebug.png', ftDebug)
# pack keypoint 2-d coords into numpy array
trackPoints1 = np.zeros((len(kp), 1, 2), dtype=np.float32)
for i, kpt in enumerate(kp):
    trackPoints1[i, :, 0] = kpt.pt[0]
    trackPoints1[i, :, 1] = kpt.pt[1]

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

trackPoints2, st, err = cv2.calcOpticalFlowPyrLK(ImT1_L, ImT2_L, trackPoints1, None, **lk_params)

ptTrackable = np.where(st == 1, 1, 0).astype(bool)
trackPoints1_KLT = trackPoints1[ptTrackable, ...]
trackPoints2_KLT_t = trackPoints2[ptTrackable, ...]
trackPoints2_KLT = np.around(trackPoints2_KLT_t)
trackPoints2_KLT

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),

                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

p0 = cv2.goodFeaturesToTrack(ImT1_L, mask=None, **feature_params)
p1, st, err = cv2.calcOpticalFlowPyrLK(ImT1_L, ImT2_L, p0, None, **lk_params)

'''
cv2.polylines(ImT1_L, np.int32(p1), True, (0,255,255), 5);
cv2.polylines(ImT2_L, np.int32(p1), True, (0,255,255), 5);

cv2.imshow('image1',ImT1_L)
cv2.waitKey(0)

cv2.imshow('image',ImT2_L)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# Create M array
M = [0] * len(p1)
for i in range(100):
    M[i] = [0] * len(p1)

# Set M[i][j] to 1 for all points i ,j in which are the same distance apart in image t+1 and image t
for i in range(0,len(p1)):
    for j in range(0,len(p1)):
        if abs((((p0[i][0][0] - p0[j][0][0])**2 + (p0[i][0][1] - p0[j][0][1])**2)**(1/2) - ((p1[i][0][0] - p1[j][0][0])**2 + (p1[i][0][1] - p1[j][0][1])**2)**(1/2))) < 0.01:
            M[i][j]=1
count = 0
maxn = 0
maxc = 0

'''
for i in range(0,len(p1)):
    for j in range(0,len(p1)):
            print ("{} ".format(M[i][j]),end='')
    print("\n")
'''

# Find point with maximum degree and store in maxn
for i in range(0,len(p1)):
    for j in range(0,len(p1)):
        if M[i][j] == 1:
            count = count+1
    if count > maxc:
        maxc = count
        maxn = i
    count=0

clique = [maxn]
isin = True

while True:
    potentialnodes = list()
    # Find potential nodes which are connected to all nodes in the clique
    for i in range(0,len(p1)):
        for j in range(0,len(clique)):
            isin = isin & M[i][clique[j]]
        if isin == True and i not in clique:
            potentialnodes.append(i)
        isin=True

    count = 0
    maxn = 0
    maxc = 0
    # Find the node which is connected to the maximum number of potential nodes and store in maxn
    for i in range(0,len(potentialnodes)):
        for j in range(0,len(potentialnodes)):
            if M[potentialnodes[i]][potentialnodes[j]] == 1:
                count = count+1
        if count > maxc:
            maxc = count
            maxn = potentialnodes[i]
        count=0
    if maxc==0:
        break
    clique.append(maxn)

