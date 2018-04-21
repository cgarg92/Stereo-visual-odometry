from math import cos, sin
import numpy as np

def genEulerZXZMatrix(psi, theta, sigma):
    # ref http://www.u.arizona.edu/~pen/ame553/Notes/Lesson%2008-A.pdf
    c1 = cos(psi)
    s1 = sin(psi)
    c2 = cos(theta)
    s2 = sin(theta)
    c3 = cos(sigma)
    s3 = sin(sigma)

    mat = np.zeros((3,3))

    mat[0,0] = (c1 * c3) - (s1 * c2 * s3)
    mat[0,1] = (-c1 * s3) - (s1 * c2 * c3)
    mat[0,2] = (s1 * s2)

    mat[1,0] = (s1 * c3) + (c1 * c2 * s3)
    mat[1,1] = (-s1 * s3) + (c1 * c2 * c3)
    mat[1,2] = (-c1 * s2)

    mat[2,0] = (s2 * s3)
    mat[2,1] = (s2 * c3)
    mat[2,2] = c2

    return mat


def minimizeReprojection(dof,d2dPoints1, d2dPoints2, d3dPoints1, d3dPoints2, w2cMatrix):
    Rmat = genEulerZXZMatrix(dof[0], dof[1], dof[2])
    translationArray = np.array([[dof[3]], [dof[4]], [dof[5]]])
    temp = np.hstack((Rmat, translationArray))
    perspectiveProj = np.vstack((temp, [0, 0, 0, 1]))
    #print (perspectiveProj)

    numPoints = d2dPoints1.shape[0]
    errorA = np.zeros((numPoints,3))
    errorB = np.zeros((numPoints,3))

    forwardProjection = np.matmul(w2cMatrix, perspectiveProj)
    backwardProjection = np.matmul(w2cMatrix, np.linalg.inv(perspectiveProj))
    for i in range(numPoints):
        Ja = np.ones((3))
        Jb = np.ones((3))
        Wa = np.ones((4))
        Wb = np.ones((4))

        Ja[0:2] = d2dPoints1[i,:]
        Jb[0:2] = d2dPoints2[i,:]
        Wa[0:3] = d3dPoints1[i,:]
        Wb[0:3] = d3dPoints2[i,:]

        JaPred = np.matmul(forwardProjection, Wb)
        JaPred /= JaPred[-1]
        e1 = Ja - JaPred

        JbPred = np.matmul(backwardProjection, Wa)
        JbPred /= JbPred[-1]
        e2 = Jb - JbPred

        errorA[i,:] = e1
        errorB[i,:] = e2

    residual = np.vstack((errorA,errorB))
    return residual.flatten()

def generate3DPoints(points2D_L, points2D_R, Proj1, Proj2):
    numPoints = points2D_L.shape[0]
    d3dPoints = np.ones((numPoints,3))

    for i in range(numPoints):
        pLeft = points2D_L[i,:]
        pRight = points2D_R[i,:]

        X = np.zeros((4,4))
        X[0,:] = pLeft[0] * Proj1[2,:] - Proj1[0,:]
        X[1,:] = pLeft[1] * Proj1[2,:] - Proj1[1,:]
        X[2,:] = pRight[0] * Proj2[2,:] - Proj2[0,:]
        X[3,:] = pRight[1] * Proj2[2,:] - Proj2[1,:]

        [u,s,v] = np.linalg.svd(X)
        v = v.transpose()
        vSmall = v[:,-1]
        vSmall /= vSmall[-1]

        d3dPoints[i, :] = vSmall[0:-1]

    return d3dPoints

'''
def genEulerZXZMatrix(psi, theta, sigma):
    mat = np.zeros((3,3))
    c1 = cos(psi)
    s1 = sin(psi)
    c2 = cos(theta)
    s2 = sin(theta)
    c3 = cos(sigma)
    s3 = sin(sigma)

    mat[0,0] = c1 * c2
    mat[0,1] = c1*s2*s3 - c3*s1
    mat[0,2] = s1*s3 + c1*c3*s2

    mat[1,0] = c2*s1
    mat[1,1] = c1*c3 + s1*s2*s3
    mat[1,2] = c3*s1*s2 - c1*s3

    mat[2,0] = -s2
    mat[2,1] = c2*s3
    mat[2,2] = c2*c3

    return mat
'''
