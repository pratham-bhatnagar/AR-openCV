import  numpy as np
import cv2

stdShape = (275,440)

webCam = cv2.VideoCapture(0)
imgTarget = cv2.imread("assets\imgTarget.jpg")
imgTarget = cv2.resize(imgTarget,stdShape)
displayVid = cv2.VideoCapture("assets\displayVid.mp4")

ORB = cv2.ORB_create(nfeatures=1000)
keyPoint1, descriptor1 = ORB.detectAndCompute(imgTarget,None)

while webCam.isOpened():
    _ , imgWebcam = webCam.read()  
    keyPoint2, descriptor2 = ORB.detectAndCompute(imgWebcam,None) 
    
    imgAR  = imgWebcam.copy()
    
    _ , imgVideo = displayVid.read()
    imgVideo = cv2.resize(imgVideo, stdShape)

    bruteForce = cv2.BFMatcher()
    matches = bruteForce.knnMatch(descriptor1,descriptor2,k=2)
    
    goodMatches = []

    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            goodMatches.append(m)
                 
    if len(goodMatches) > 15:
        srcPts = np.float32([keyPoint1[m.queryIdx].pt for m in goodMatches]).reshape(-1,1,2)
        dstPts = np.float32([keyPoint2[m.trainIdx].pt for m in goodMatches]).reshape(-1,1,2)
        matrix , mask = cv2.findHomography(srcPts,dstPts,cv2.RANSAC,5)
        
        pts = np.float32([[0,0],[0,440],[275,440],[275,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,matrix)
        cv2.polylines(imgWebcam,[np.int32(dst)],True,(0,0,255),3)
        cv2.imshow("Polylines",imgWebcam)
        imgWarp = cv2.warpPerspective(imgVideo,matrix,(imgWebcam.shape[1],imgWebcam.shape[0]))
        
        newmask = np.zeros((imgWebcam.shape[0],imgWebcam.shape[1]),np.uint8)
        cv2.fillPoly(newmask,[np.int32(dst)],(255,255,255))
        
        invMask = cv2.bitwise_not(newmask)
        imgAR = cv2.bitwise_and(imgAR,imgAR,mask=invMask)
        imgAR = cv2.bitwise_or(imgAR,imgWarp)
    
    cv2.imshow("imgAR",imgAR)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break

webCam.release()

# Code by Pratham Bhatnagar.