import cv2
import numpy as np
import os

def saveDebugImg(imgIn, frmId, tag, points, color=None, postTag=''):
    if not os.path.exists('debugImgs'):
        os.makedirs('debugImgs')

    imgD = imgIn.copy()
    if isinstance(points, list):
        imgD = cv2.drawKeypoints(imgIn, points, imgD, color=color)
    else:
        imgD = cv2.cvtColor(imgD,cv2.COLOR_GRAY2RGB)
        for point in points:
            cv2.circle(imgD, (point[0], point[1]), 2, color=color)

    outFileName = 'debugImgs/' + tag + '_' + str(frmId)
    if postTag != '':
        outFileName = outFileName + '_' + postTag

    outFileName = outFileName + '.png'
    cv2.imwrite(outFileName, imgD)
    return
