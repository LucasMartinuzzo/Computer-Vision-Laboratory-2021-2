import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GLUT import *

RESIZE = 0.4
TEXTURE_SIZE = 512

def drawimage(img, px, py):
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    h,w,_ = img.shape
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0.0, w, 0.0, h, 0.0, 200.0)
    glRasterPos2i(px, py)
    glDepthMask(GL_FALSE)
    glDrawPixels(w,h, GL_RGB, GL_UNSIGNED_BYTE,
                 np.fliplr(img).tobytes()[::-1])
    glDepthMask(GL_TRUE)    
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

def getimage(SizeX, SizeY):
    glReadBuffer( GL_FRONT );
    im = glReadPixels(0,0, SizeX, SizeY,
                      GL_RGB, GL_UNSIGNED_BYTE)
    t1 = np.copy(np.frombuffer(im, np.uint8)[::-1])
    t2 = t1.reshape(SizeY, SizeX, 3)
    t3 = np.fliplr(t2)
    return t3

def drawcube(d,tex_side_1,tex_side_2,tex_top_bottom):
    #red = [1., 0., 0., 1.]
    #glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, red)
    white = [1., 1., 1., 1.]
    glMaterialfv(GL_FRONT, GL_SPECULAR, white)
    glMaterialfv(GL_FRONT, GL_SHININESS, 128)
    glPolygonMode(GL_FRONT, GL_FILL) #FILL)
    glPolygonMode(GL_BACK, GL_FILL)    
    #glColor3f(1.0, 0.0, 0.0)
    L = [(-1,-1), (1,-1), (1,1), (-1,1)]
    for k in range(-1,2,2): #k = -1,1
        L.reverse()
        red = [1., 0., 0., 1.]
        glBindTexture(GL_TEXTURE_2D,tex_top_bottom) #TOP
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, white)
        glBegin(GL_QUADS)
        glNormal3f(0.0, 0.0, k)
        for (j,i) in L:
            glTexCoord2f((j+1)/2, (i+1)/2)
            glVertex3f(j*d, i*d, k*d)
        glEnd()
        green = [0., 1., 0., 1.]
        glBindTexture(GL_TEXTURE_2D,tex_side_1)
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, white)
        glBegin(GL_QUADS)
        glNormal3f(k, 0.0, 0.0)
        for (j,i) in L:
            glTexCoord2f((j+1)/2, (i+1)/2)
            glVertex3f(k*d, j*d, i*d)
        glEnd()        
        blue = [0., 0., 1., 1.]
        glBindTexture(GL_TEXTURE_2D,tex_side_2)
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, white)
        glBegin(GL_QUADS)
        glNormal3f(0.0, k, 0.0)
        for (j,i) in L:
            glTexCoord2f((j+1)/2, (i+1)/2)
            glVertex3f(i*d, k*d, j*d)
        glEnd()
        

def showScreen(tex,image,Trans,arucoL):
    glDepthMask(GL_TRUE)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #glLoadIdentity()
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_LIGHTING)
    drawimage(image, 0, 0)
    glEnable(GL_TEXTURE_2D) 
    glEnable(GL_LIGHTING)

    glLoadMatrixd(np.transpose(Trans).flatten())
    desloc = -2*arucoL/6
    #Base
    glTranslatef(0, 0, desloc/2)
    drawcube(arucoL/6,tex[0],tex[2],tex[1])
    #Altura 1
    glTranslatef(0, 0, desloc)
    drawcube(arucoL/6,tex[0],tex[2],tex[1])
    #Altura 2
    glTranslatef(0, 0, desloc)
    drawcube(arucoL/6,tex[0],tex[2],tex[1])
    #Folha Altura 3
    glTranslatef(0, 0, desloc)
    drawcube(arucoL/6,tex[3],tex[3],tex[3])
    #Folha Altura 2 - atras
    glTranslatef(0, -desloc, -desloc)
    drawcube(arucoL/6,tex[3],tex[3],tex[3])
    #Folha Altura 2 - atras esquerda
    glTranslatef(-desloc, 0, 0)
    drawcube(arucoL/6,tex[3],tex[3],tex[3])
    #Folha Altura 2 - atras direita
    glTranslatef(2*desloc, 0, 0)
    drawcube(arucoL/6,tex[3],tex[3],tex[3])
    #Folha Altura 2 - meio direita
    glTranslatef(0, desloc, 0)
    drawcube(arucoL/6,tex[3],tex[3],tex[3])
    #Folha Altura 2 - meio esquerda
    glTranslatef(-2*desloc, 0, 0)
    drawcube(arucoL/6,tex[3],tex[3],tex[3])
    #Folha Pequena Altura 3 - meio esquerda
    glTranslatef(2*desloc, 0, 0)
    glTranslatef(-desloc/4, desloc/4, 3*desloc/4)
    drawcube(arucoL/12,tex[3],tex[3],tex[3])
    glTranslatef(0, -desloc/2, 0)
    drawcube(arucoL/12,tex[3],tex[3],tex[3])
    #Folha Pequena Altura 3 - atras esquerda
    glTranslatef(0, -desloc/2, 0)
    drawcube(arucoL/12,tex[3],tex[3],tex[3])
    #Folha Pequena Altura 3 - atras meio 1
    glTranslatef(-desloc/2, 0, 0)
    drawcube(arucoL/12,tex[3],tex[3],tex[3])
    #Folha Pequena Altura 3 - atras meio 2
    glTranslatef(-desloc/2, 0, 0)
    drawcube(arucoL/12,tex[3],tex[3],tex[3])
    #Folha Pequena Altura 3 - atras esquerda 1
    glTranslatef(-desloc/2, 0, 0)
    drawcube(arucoL/12,tex[3],tex[3],tex[3])
    #Folha Pequena Altura 3 - meio esquerda 1
    glTranslatef(0, desloc/2, 0)
    drawcube(arucoL/12,tex[3],tex[3],tex[3])
    #Folha Pequena Altura 3 - meio esquerda 2
    glTranslatef(0, desloc/2, 0)
    drawcube(arucoL/12,tex[3],tex[3],tex[3])
    #Folha Pequena Altura 3 - frente esquerda 1
    glTranslatef(0, desloc/2, 0)
    drawcube(arucoL/12,tex[3],tex[3],tex[3])
    #Folha Pequena Altura 3 - frente meio 1
    glTranslatef(desloc/2, 0, 0)
    drawcube(arucoL/12,tex[3],tex[3],tex[3])
    #Folha Pequena Altura 3 - frente meio 2
    glTranslatef(desloc/2, 0, 0)
    drawcube(arucoL/12,tex[3],tex[3],tex[3])
    #Folha Pequena Altura 3 - frente meio 3
    glTranslatef(desloc/2, 0, 0)
    drawcube(arucoL/12,tex[3],tex[3],tex[3])
    #Folha Pequena Altura 4 - meio
    glTranslatef(-3*desloc/4, -3*desloc/4, desloc)
    drawcube(arucoL/24,tex[3],tex[3],tex[3])
    glutSwapBuffers()


def init(SizeX, SizeY,mtx):
    far = 100.0
    near = 0.01
    glViewport(0,0, SizeX,SizeY)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    #           l    r     b    t    n    f
    #glFrustum(-250, 250, -250, 250, 200, 500)
    Proj = np.zeros((4,4))
    Proj[0,0] = 2.0*mtx[0,0]/SizeX
    Proj[1,1] = 2.0*mtx[1,1]/SizeY
    Proj[0,2] =  ( 1.0 - (2.0*mtx[0,2])/SizeX)
    Proj[1,2] = -(1.0 - (2.0*mtx[1,2])/SizeY)
    Proj[2,2] = -(far+near)/(far-near)
    Proj[2,3] = -2.0*far*near/(far-near)
    Proj[3,2] = -1.0
    glLoadMatrixd(np.transpose(Proj).flatten())
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


MINCONTOUR = 200
def main():
    P0 = [[1,0,1,1],
          [0,1,0,1],
          [0,0,1,1],
          [0,0,1,0]]
    P1 = [[0,0,0,0],
          [1,1,1,1],
          [1,0,0,1],
          [1,0,1,0]]
    P2 = [[0,0,1,1],
          [0,0,1,1],
          [0,0,1,0],
          [1,1,0,1]]
    P3 = [[1,0,0,1],
          [1,0,0,1],
          [0,1,0,0],
          [0,1,1,0]]
    P = [np.array(P0),
         np.array(P1),
         np.array(P2),
         np.array(P3)]
    Sol = [None]*4
    video = "aruco.mp4"
    video_path = os.path.join("./videos",video)
    if not os.path.isfile(video_path):
        print("File not found. Make sure the videos are in the folder ./videos",
              "with the correct name and the format .mp4.")
        return -1
    capture = cv2.VideoCapture(video_path)
    retval,frame = capture.read()
    image = cv2.resize(frame, None, fx=RESIZE, fy=RESIZE)
    SizeY, SizeX, _ = image.shape
    print("Image size:",SizeY,SizeX)
    
    
    mtx = np.loadtxt("mtx.txt")
    dist = np.loadtxt("dist.txt")
    dist = dist.reshape(1,5)
    
    
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(SizeX, SizeY)
    glutInitWindowPosition(0, 0)
    wind = glutCreateWindow("OpenGL")
    
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [0,0,0,1] )
    glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.5, 0.5, 0.5, 1.])
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.7, 0.7, 0.7, 1.])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.])
    glLightfv(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.00002)
    
    glEnable(GL_TEXTURE_2D)
    tex = glGenTextures(5)
    glBindTexture(GL_TEXTURE_2D, tex[0])
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MAG_FILTER,
                    GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR)
    img = cv2.imread("texture/tree_trunk.png")
    glTexImage2D(GL_TEXTURE_2D, 0,
                 GL_RGB, TEXTURE_SIZE, TEXTURE_SIZE, 0,
                 GL_RGB, GL_UNSIGNED_BYTE,
                 img.tobytes()[::-1])
    
    glBindTexture(GL_TEXTURE_2D, tex[1])
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MAG_FILTER,
                    GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR)
    img = cv2.imread("texture/tree_top.png")
    glTexImage2D(GL_TEXTURE_2D, 0,
                 GL_RGB, TEXTURE_SIZE, TEXTURE_SIZE, 0,
                 GL_RGB, GL_UNSIGNED_BYTE,
                 img.tobytes()[::-1])
    
    glBindTexture(GL_TEXTURE_2D, tex[2])
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MAG_FILTER,
                    GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR)
    img = cv2.imread("texture/tree_trunk2.png")
    glTexImage2D(GL_TEXTURE_2D, 0,
                 GL_RGB, TEXTURE_SIZE, TEXTURE_SIZE, 0,
                 GL_RGB, GL_UNSIGNED_BYTE,
                 img.tobytes()[::-1])
    
    glBindTexture(GL_TEXTURE_2D, tex[3])
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MAG_FILTER,
                    GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR)
    img = cv2.imread("texture/leaves2.png")
    glTexImage2D(GL_TEXTURE_2D, 0,
                 GL_RGB, TEXTURE_SIZE, TEXTURE_SIZE, 0,
                 GL_RGB, GL_UNSIGNED_BYTE,
                 img.tobytes()[::-1])
    
    glBindTexture(GL_TEXTURE_2D, tex[4])
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MAG_FILTER,
                    GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR)
    img = cv2.imread("texture/grass.png")
    glTexImage2D(GL_TEXTURE_2D, 0,
                 GL_RGB, TEXTURE_SIZE, TEXTURE_SIZE, 0,
                 GL_RGB, GL_UNSIGNED_BYTE,
                 img.tobytes()[::-1])
    
    glutDisplayFunc(showScreen)
    glutIdleFunc(showScreen)
    init(SizeX, SizeY,mtx)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter('output.avi', fourcc, 30.0, (SizeX, SizeY))
    while True:
        Sol = [None]*4
        retval,frame = capture.read()
        if not retval:
            break
        image = cv2.resize(frame, None, fx=RESIZE, fy=RESIZE)
        SizeY, SizeX, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.ADAPTIVE_THRESH_MEAN_C
        mask = cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 71, 7)
        #cv2.imwrite("mask.png", mask)
        contours,_ = cv2.findContours(mask,
                                      cv2.RETR_LIST, #cv2.RETR_TREE
                                      cv2.CHAIN_APPROX_NONE)
        
        #print(len(contours))
        
        contours2 = []
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > MINCONTOUR:
                contours2.append(cnt)
        
        #print(len(contours2))
        
        contours3 = []
        for cnt in contours2:
            eps = cv2.arcLength(cnt, True)*0.05
            approx = cv2.approxPolyDP(cnt,eps,True)
            if len(approx) != 4:
                continue
            if cv2.isContourConvex(approx):
                contours3.append(approx)
        
        #print(len(contours3))
        
        for cnt in contours3:
            v01 = [cnt[1,0,0]-cnt[0,0,0], cnt[1,0,1]-cnt[0,0,1]]
            v02 = [cnt[2,0,0]-cnt[0,0,0], cnt[2,0,1]-cnt[0,0,1]]
            pv = np.cross(v01, v02)
            if pv < 0:
                cnt[1,0,0],cnt[3,0,0] = cnt[3,0,0],cnt[1,0,0]
                cnt[1,0,1],cnt[3,0,1] = cnt[3,0,1],cnt[1,0,1]
        
        Lc = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]
        for cnt in contours3:
            pts = []
            for i in range(4):
                x = cnt[i,0,0]
                y = cnt[i,0,1]
                pts.append([x,y])       
            W = H = 300
            input_pts = np.float32(pts)
            output_pts = np.float32([[  0,   0],
                                     [W-1,   0],
                                     [W-1, H-1],
                                     [  0, H-1]])
            M = cv2.getPerspectiveTransform(input_pts, output_pts)
            out = cv2.warpPerspective(gray, M,
                                     (W,H),
                                     flags=cv2.INTER_LINEAR)
            (T,thresh) = cv2.threshold(out, 0, 255,
                                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            dw = W/6
            dh = H/6
            C = []
            y = dh
            for i in range(4):
                x = dw
                linha = []
                for j in range(4):
                    x1,y1 = round(x),round(y)
                    x2,y2 = round(x+dw),round(y+dh)
                    cell = thresh[y1:y2, x1:x2]
                    c = cv2.countNonZero(cell)
                    if c > dw*dh/2:
                        linha.append(1)
                    else:
                        linha.append(0)
                    x += dw
                C.append(linha)
                y += dh
            #print(C)
            ropt = IDopt = -1
            cod = np.array(C)
            for r in range(4):
                for ID in range(len(P)):
                    if (cod == P[ID]).all():
                        ropt = r
                        IDopt = ID
                cod = np.rot90(cod, k=1)
            #print(ropt,IDopt)
            #cv2.imshow("out", thresh)
            #cv2.waitKey(0)
            if IDopt == -1:
                continue
            input_pts = np.roll(input_pts, shift=-ropt, axis=0)
            Sol[IDopt] = input_pts
        
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for k in range(len(Sol)):
            if Sol[k] is None:
                continue
            Sol[k] = cv2.cornerSubPix(gray, Sol[k],
                                      (11,11), (-1,-1),
                                      criteria)
            
            
        #PARA MOSTRAR OS PONTOS DO ARUCO, DESCOMENTAR AQUI:
        # for sol in Sol:
        #     if sol is None:
        #         continue
        #     for i in range(4):
        #         x,y = int(sol[i,0]),int(sol[i,1])
        #         cv2.circle(image,(x,y),16, Lc[i], -1)        
        #-----------------------
    
        #print(dist.shape)
        
        arucoL = 15.4 #cm
        
        p2D = Sol[0]
        p3D = np.zeros((4,3), np.float32)
        k = 0
        for i in range(-1,2,2): #i = -1,1
            for j in range(-1,2,2): #j = -1,1
                p3D[k,0] = j*arucoL/2
                p3D[k,1] = i*arucoL/2
                k += 1
        p3D[2,0],p3D[3,0] = p3D[3,0],p3D[2,0]
        p3D[2,1],p3D[3,1] = p3D[3,1],p3D[2,1]
        #print(p3D)
        
        try: 
            success,rv,tv = cv2.solvePnP(p3D,
                                     p2D,
                                     mtx,
                                     dist,
                                     flags=0)
        except:
            continue
        #Rotação em 180 em relação ao eixo X
        Rx = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        
        R,_ = cv2.Rodrigues(rv)
        #print(R)
        RT = np.transpose(R)
        Trans = np.zeros((4,4))
        Trans[:3,:3] = Rx@R
        Trans[:3,3:] = Rx@tv #-(RT@tv)
        Trans[3,3] = 1.0
        #print(Trans)
    
    
        
        
        showScreen(tex,image,Trans,arucoL)
        #glutMainLoop()
        #-----------------------
        #cv2.imwrite("detected.png", image)
        cv2.waitKey(2)
        cv2.destroyAllWindows()
        
        imageGL = getimage(SizeX, SizeY)
        #cv2.imwrite("out.png", imageGL)
        out_video.write(imageGL)
    
    out_video.release()
    glutHideWindow()
    glutDestroyWindow(wind)
    glutMainLoopEvent()
    print('End')
    return 0

if __name__ == "__main__":
    main()

