import numpy as np
import cv2

squareL = 23/9 #cm
NUM_IMAGES = 32
RESIZE = 0.4
objp = np.zeros((9*6,3), np.float32)
k = 0
for i in range(6):
    for j in range(9):
        objp[k,0] = j #*squareL
        objp[k,1] = i #*squareL
        k += 1
#print(objp)

criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

Lc = []
Lw = []
for i in range(NUM_IMAGES):
    img = cv2.imread("xadrez/cam%02d.png"%(i+1))
    img = cv2.resize(img, None, fx=RESIZE, fy=RESIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret,corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners,
                                    (11,11), (-1,-1),
                                    criteria)
        Lc.append(corners2)
        Lw.append(objp)
        cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        cv2.imwrite("out/corners%02d.png"%(i+1), img)
h,w = gray.shape
ret,mtx,dist,rv,tv = cv2.calibrateCamera(Lw, Lc, (w,h), None, None)

print(mtx)
print(dist)

np.savetxt("mtx.txt", mtx)
np.savetxt("dist.txt", dist)

mean_error = 0
for i in range(len(Lw)):
    imgp,_ = cv2.projectPoints(Lw[i], rv[i], tv[i], mtx, dist)
    error = cv2.norm(imgp, Lc[i], cv2.NORM_L2)/len(imgp)
    mean_error += error
print("error:",mean_error/len(Lw))
#------------------------------------

for k in range(NUM_IMAGES):
    img = cv2.imread("xadrez/cam%02d.png"%(k+1))
    img = cv2.resize(img, None, fx=RESIZE, fy=RESIZE)
    L = []
    for i in range(2,5,2): #i = 2,4
        for j in range(2,5,2): #j = 2,4
            P = np.array([ [j,i,0], [j,i,-3] ], dtype=np.float32)
            p,_ = cv2.projectPoints(P, rv[k], tv[k], mtx, dist)
            x1,y1 = int(p[0,0,0]), int(p[0,0,1])
            x2,y2 = int(p[1,0,0]), int(p[1,0,1])
            cv2.circle(img, (x1,y1), 15, (0,0,255), -1)
            cv2.circle(img, (x2,y2), 15, (0,255,0), -1)
            cv2.line(img, (x1,y1), (x2,y2), (0,255,255), 7)
            L.append((x2,y2))
    L[2],L[3] = L[3],L[2]
    for i in range(len(L)): #i = 0,1,2,3
        cv2.line(img, L[i-1], L[i], (255,0,255), 7)

    cv2.imwrite("out/img%02d.png"%(k+1), img)
#------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()
