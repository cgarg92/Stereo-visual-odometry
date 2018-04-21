import numpy as np
from matplotlib import pyplot as plt
import cv2
import sys
from scipy.optimize import least_squares
import os
import inlierDetector
from helperFunctions import genEulerZXZMatrix, minimizeReprojection
from utils import saveDebugImg

if __name__ == "__main__":

    sequence = 00   #sys.argv[1]
    startFrame = 0 #sys.argv[2]
    endFrame = 10 #sys.argv[3]
    plotTrajectory = False
    outputDebug = False

    datapath = '../Data/' + '{0:02d}'.format(sequence)

    calibFileName = datapath + '/calib.txt'
    calibFile = open(calibFileName, 'r').readlines()
    P1Vals = calibFile[0].split()
    Proj1 = np.zeros((3,4))
    for row in range(3):
        for column in range(4):
            Proj1[row, column] = float(P1Vals[row*4 + column + 1])

    P2Vals = calibFile[1].split()
    Proj2 = np.zeros((3,4))
    for row in range(3):
        for column in range(4):
            Proj2[row, column] = float(P2Vals[row*4 + column + 1])

    leftImagePath = datapath + '/image_0/'
    rightImagePath = datapath + '/image_1/'

    translation = None
    rotation = None

    fpPoseOut = open('svoPoseOut.txt', 'wb')

    traj = np.zeros((600,600,3), dtype=np.uint8)

    for frm in range(startFrame+1, endFrame+1):

        # reuse T-1 data instead of reading again-again
        # same with feature computation - anything that can be reused
        imgPath = leftImagePath + '{0:06d}'.format(frm-1) + '.png';
        ImT1_L = cv2.imread(imgPath, 0)    #0 flag returns a grayscale image

        imgPath = rightImagePath + '{0:06d}'.format(frm-1) + '.png';
        ImT1_R = cv2.imread(imgPath, 0)

        imgPath = leftImagePath + '{0:06d}'.format(frm) + '.png';
        ImT2_L = cv2.imread(imgPath, 0)

        imgPath = rightImagePath + '{0:06d}'.format(frm) + '.png';
        ImT2_R = cv2.imread(imgPath, 0)

        block = 15
        #emperical values from P1, P2 as suggested in Ocv documentation
        P1 = block * block * 8
        P2 = block * block * 32

        disparityEngine = cv2.StereoSGBM_create(minDisparity=0,numDisparities=16, blockSize=block, P1=P1, P2=P2)
        ImT1_disparity = disparityEngine.compute(ImT1_L, ImT1_R).astype(np.float32)
        #cv2.imwrite('disparity.png', ImT1_disparity)
        ImT1_disparityA = np.divide(ImT1_disparity, 16.0)

        ImT2_disparity = disparityEngine.compute(ImT2_L, ImT2_R).astype(np.float32)
        ImT2_disparityA = np.divide(ImT2_disparity, 16.0)

        if outputDebug:
            fname = 'debugImgs/diparity_' + str(frm) + '.png'
            cv2.imwrite(fname, ImT2_disparityA)

        TILE_H = 10
        TILE_W = 20
        fastFeatureEngine = cv2.FastFeatureDetector_create()

        #20x10 (wxh) tiles for extracting less features from images
        H,W = ImT1_L.shape
        kp = []
        idx = 0
        for y in range(0, H, TILE_H):
            for x in range(0, W, TILE_W):
                imPatch = ImT1_L[y:y+TILE_H, x:x+TILE_W]
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

        if outputDebug:
            saveDebugImg(ImT1_L, frm-1, 'keypoints', kp, color=(255,0,0))

        # pack keypoint 2-d coords into numpy array
        trackPoints1 = cv2.KeyPoint_convert(kp)
        trackPoints1 = np.expand_dims(trackPoints1, axis=1)

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 3,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

        trackPoints2, st, err = cv2.calcOpticalFlowPyrLK(ImT1_L, ImT2_L, trackPoints1, None, flags=cv2.MOTION_AFFINE, **lk_params)

        # separate points that were tracked successfully
        ptTrackable = np.where(st == 1, 1,0).astype(bool)
        trackPoints1_KLT = trackPoints1[ptTrackable, ...]
        trackPoints2_KLT_t = trackPoints2[ptTrackable, ...]
        trackPoints2_KLT = np.around(trackPoints2_KLT_t)

        # among tracked points take points within error measue
        error = 4
        errTrackablePoints = err[ptTrackable, ...]
        errThresholdedPoints = np.where(errTrackablePoints < error, 1, 0).astype(bool)
        trackPoints1_KLT = trackPoints1_KLT[errThresholdedPoints, ...]
        trackPoints2_KLT = trackPoints2_KLT[errThresholdedPoints, ...]

        if outputDebug:
            saveDebugImg(ImT1_L, frm, 'trackedPt', trackPoints1_KLT, color=(0,255,255), postTag='0')
            saveDebugImg(ImT2_L, frm, 'trackedPt', trackPoints2_KLT, color=(0,255,0), postTag='1')

        # check for validity of tracked point Coordinates
        hPts = np.where(trackPoints2_KLT[:,1] >= H)
        wPts = np.where(trackPoints2_KLT[:,0] >= W)
        outTrackPts = hPts[0].tolist() + wPts[0].tolist()
        outDeletePts = list(set(outTrackPts))

        if len(outDeletePts) > 0:
            trackPoints1_KLT_L = np.delete(trackPoints1_KLT, outDeletePts, axis=0)
            trackPoints2_KLT_L = np.delete(trackPoints2_KLT, outDeletePts, axis=0)
        else:
            trackPoints1_KLT_L = trackPoints1_KLT
            trackPoints2_KLT_L = trackPoints2_KLT

        #compute right image disparity displaced points
        trackPoints1_KLT_R = np.copy(trackPoints1_KLT_L)
        trackPoints2_KLT_R = np.copy(trackPoints2_KLT_L)
        selectedPointMap = np.zeros(trackPoints1_KLT_L.shape[0])

        disparityMinThres = 0.0
        disparityMaxThres = 100.0

        for i in range(trackPoints1_KLT_L.shape[0]):
            T1Disparity = ImT1_disparityA[int(trackPoints1_KLT_L[i,1]), int(trackPoints1_KLT_L[i,0])]
            T2Disparity = ImT2_disparityA[int(trackPoints2_KLT_L[i,1]), int(trackPoints2_KLT_L[i,0])]
            # try:
            #     T2Disparity = ImT2_disparityA[int(trackPoints2_KLT_L[i,1]), int(trackPoints2_KLT_L[i,0])]
            # except:
            #     print (int(trackPoints2_KLT_L[i,1]), int(trackPoints2_KLT_L[i,0]))

            if (T1Disparity > disparityMinThres and T1Disparity < disparityMaxThres
                and T2Disparity > disparityMinThres and T2Disparity < disparityMaxThres):
                trackPoints1_KLT_R[i, 0] = trackPoints1_KLT_L[i, 0] - T1Disparity
                trackPoints2_KLT_R[i, 0] = trackPoints2_KLT_L[i, 0] - T2Disparity
                selectedPointMap[i] = 1

        selectedPointMap = selectedPointMap.astype(bool)
        trackPoints1_KLT_L_3d = trackPoints1_KLT_L[selectedPointMap, ...]
        trackPoints1_KLT_R_3d = trackPoints1_KLT_R[selectedPointMap, ...]
        trackPoints2_KLT_L_3d = trackPoints2_KLT_L[selectedPointMap, ...]
        trackPoints2_KLT_R_3d = trackPoints2_KLT_R[selectedPointMap, ...]

        # 3d point cloud triagulation

        numPoints = trackPoints1_KLT_L_3d.shape[0]
        d3dPointsT1 = np.ones((numPoints,3))
        d3dPointsT2 = np.ones((numPoints,3))

        for i in range(numPoints):
            #for i in range(1):
            pLeft = trackPoints1_KLT_L_3d[i,:]
            pRight = trackPoints1_KLT_R_3d[i,:]

            X = np.zeros((4,4))
            X[0,:] = pLeft[0] * Proj1[2,:] - Proj1[0,:]
            X[1,:] = pLeft[1] * Proj1[2,:] - Proj1[1,:]
            X[2,:] = pRight[0] * Proj2[2,:] - Proj2[0,:]
            X[3,:] = pRight[1] * Proj2[2,:] - Proj2[1,:]

            [u,s,v] = np.linalg.svd(X)
            v = v.transpose()
            vSmall = v[:,-1]
            vSmall /= vSmall[-1]

            d3dPointsT1[i, :] = vSmall[0:-1]

        for i in range(numPoints):
            #for i in range(1):
            pLeft = trackPoints2_KLT_L_3d[i,:]
            pRight = trackPoints2_KLT_R_3d[i,:]

            X = np.zeros((4,4))
            X[0,:] = pLeft[0] * Proj1[2,:] - Proj1[0,:]
            X[1,:] = pLeft[1] * Proj1[2,:] - Proj1[1,:]
            X[2,:] = pRight[0] * Proj2[2,:] - Proj2[0,:]
            X[3,:] = pRight[1] * Proj2[2,:] - Proj2[1,:]

            [u,s,v] = np.linalg.svd(X)
            v = v.transpose()
            vSmall = v[:,-1]
            vSmall /= vSmall[-1]

            d3dPointsT2[i, :] = vSmall[0:-1]

        #tunable - def 0.01
        distDifference = 0.1

        # in-lier detection algorithm
        clique = inlierDetector.findClique(d3dPointsT1, d3dPointsT2, distDifference)

        # pick up clique point 3D coords and features for optimization
        pointsInClique = len(clique)
        cliqued3dPointT1 = d3dPointsT1[clique]#np.zeros((pointsInClique, 3))
        cliqued3dPointT2 = d3dPointsT2[clique]

        # points = features
        trackedPoints1_KLT_L = trackPoints1_KLT_L_3d[clique]
        trackedPoints2_KLT_L = trackPoints2_KLT_L_3d[clique]

        if outputDebug:
            saveDebugImg(ImT1_L, frm, 'clique', trackedPoints1_KLT_L, color=(0,255,255), postTag='0')
            saveDebugImg(ImT2_L, frm, 'clique', trackedPoints2_KLT_L, color=(0,255,0), postTag='1')


        if (trackedPoints1_KLT_L.shape[0] < 6):
            continue
        dSeed = np.zeros(6)
        #minimizeReprojection(d, trackedPoints1_KLT_L, trackedPoints2_KLT_L, cliqued3dPointT1, cliqued3dPointT2, Proj1)
        optRes = least_squares(minimizeReprojection, dSeed, method='lm', max_nfev=200,
                            args=(trackedPoints1_KLT_L, trackedPoints2_KLT_L, cliqued3dPointT1, cliqued3dPointT2, Proj1))

        error = optRes.fun
        pointsInClique = len(clique)
        e = error.reshape((pointsInClique*2, 3))
        errorThreshold = 0.5
        xRes1 = np.where(e[0:pointsInClique, 0] >= errorThreshold)
        yRes1 = np.where(e[0:pointsInClique, 1] >= errorThreshold)
        zRes1 = np.where(e[0:pointsInClique, 2] >= errorThreshold)
        xRes2 = np.where(e[pointsInClique:2*pointsInClique, 0] >= errorThreshold)
        yRes2 = np.where(e[pointsInClique:2*pointsInClique, 1] >= errorThreshold)
        zRes2 = np.where(e[pointsInClique:2*pointsInClique, 2] >= errorThreshold)

        pruneIdx = xRes1[0].tolist() + yRes1[0].tolist() + zRes1[0].tolist() + xRes2[0].tolist() + yRes2[0].tolist() +  zRes2[0].tolist()
        if (len(pruneIdx) > 0):
            uPruneIdx = list(set(pruneIdx))
            trackedPoints1_KLT_L = np.delete(trackedPoints1_KLT_L, uPruneIdx, axis=0)
            trackedPoints2_KLT_L = np.delete(trackedPoints2_KLT_L, uPruneIdx, axis=0)
            cliqued3dPointT1 = np.delete(cliqued3dPointT1, uPruneIdx, axis=0)
            cliqued3dPointT2 = np.delete(cliqued3dPointT2, uPruneIdx, axis=0)

            if (trackedPoints1_KLT_L.shape[0] >= 6):
                optRes = least_squares(minimizeReprojection, optRes.x, method='lm', max_nfev=200,
                            args=(trackedPoints1_KLT_L, trackedPoints2_KLT_L, cliqued3dPointT1, cliqued3dPointT2, Proj1))

        if outputDebug:
            saveDebugImg(ImT2_L, frm, 'cliqueReProjSelect', trackedPoints1_KLT_L, color=(0,255,0))
        #clique size check
        # reproj error check
        # r, t generation
        Rmat = genEulerZXZMatrix(optRes.x[0], optRes.x[1], optRes.x[2])
        translationArray = np.array([[optRes.x[3]], [optRes.x[4]], [optRes.x[5]]])

        if (isinstance(translation, np.ndarray)):
            translation = translation + np.matmul(rotation, translationArray)
        else:
            translation = translationArray

        if (isinstance(rotation, np.ndarray)):
            rotation = np.matmul(Rmat, rotation)
        else:
            rotation = Rmat

        outMat = np.hstack((rotation, translation))
        np.savetxt(fpPoseOut, outMat, fmt='%.6e', footer='\n')

        #print (outMat)
        #print ()

        if plotTrajectory:
            draw_x, draw_y = int(translation[0])+290, int(translation[2])+90
            cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
            text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(translation[0],translation[1],translation[2])
            cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
            cv2.circle(traj, (draw_x, draw_y), 1, (frm*255/(endFrame-startFrame),255-frm*255/(endFrame-startFrame),0), 1)
            cv2.imshow('Trajectory', traj)
            cv2.waitKey(1)

        if frm % 10 == 0:
            print (frm)
    cv2.imwrite('map.png', traj)
