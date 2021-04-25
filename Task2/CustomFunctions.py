import numpy as np
import cv2 as cv2
from scipy import signal, ndimage
import os

def normalize(X:np.array):
    a = np.min(X)
    b = np.max(X)
    return 256*(X-a)/(b-a)

def SSD(X:np.array,Y:np.array):
    return np.sum(np.square(X-Y))

def read_data_from_disk():
	faces = []
	target = []
	folder = os.path.dirname(os.path.abspath(__file__)) + "/orl_faces/s"
	for i in range(1, 41):
		for j in range(1, 11):
			image = cv2.cvtColor(cv2.imread(folder + str(i) + "/" + str(j) + ".pgm"), cv2.COLOR_BGR2GRAY)
			faces.append(image)#/255)
			target.append(i)
	return [np.array(faces, dtype='single'), np.array(target)]

#def CrossCorrelationBasic(X:np.array,Y:np.array):
#    return np.sum(X*Y)

def TemplateMatchingBasic(X:np.array,kernel:np.array):
    MinValue = 2**64
    minpos = []
    Res = np.empty([X.shape[0]-kernel.shape[0]+1,X.shape[1]-kernel.shape[1]+1])
    for x in range(0,X.shape[0]-kernel.shape[0]+1):
        for y in range(0,X.shape[1]-kernel.shape[1]+1):
            pos = np.array([x,y])
            #Value = SSD(X[pos[0]:pos[0]+ kernel.shape[0],pos[1]:pos[1]+ kernel.shape[1]],kernel)
            #Value = crossCorrelationBasic(X[pos[0]:pos[0]+ kernel.shape[0],pos[1]:pos[1]+ kernel.shape[1]],kernel)
            Value = np.sum(np.square(X[pos[0]:pos[0]+ kernel.shape[0],pos[1]:pos[1]+ kernel.shape[1]] - kernel))
            Res[x,y]=Value
            if Value <= MinValue:
                minpos = pos
                MinValue = Value
    #print(MinValue," ",minpos)
    return Res, minpos

def CrossCorrelationBasic(X:np.array,kernel:np.array):
    MaxValue = 0
    maxpos = []
    Res = np.empty([X.shape[0]-kernel.shape[0]+1,X.shape[1]-kernel.shape[1]+1])
    for x in range(0,X.shape[0]-kernel.shape[0]+1):
        for y in range(0,X.shape[1]-kernel.shape[1]+1):
            pos = np.array([x,y])
            #Value = SSD(X[pos[0]:pos[0]+ kernel.shape[0],pos[1]:pos[1]+ kernel.shape[1]],kernel)
            Value = np.sum(X[pos[0]:pos[0]+ kernel.shape[0],pos[1]:pos[1]+ kernel.shape[1]]*kernel)
            #Value = np.sum(np.square(X[pos[0]:pos[0]+ kernel.shape[0],pos[1]:pos[1]+ kernel.shape[1]] - kernel))
            Res[x,y]=Value
            if Value >= MaxValue:
                maxpos = pos
                MaxValue = Value
    #print(MaxValue," ",maxpos)
    return Res, maxpos

def CrossCorrelationFourier(X:np.array,kernel:np.array):
    #kernelF = np.fft.fft2(kernel)
    #XF = np.fft.fft2(X)
    #Res = XF*kernelF
    #Res = np.fft.iftt2(Res)
    Res = signal.fftconvolve(X, np.flip(kernel), mode='valid') #mode='same'
    ind = np.unravel_index(np.argmax(Res, axis=None), Res.shape)
    Value = Res[ind]
    #print(Value," ",ind)
    return Res, ind

def CrossCorrelationMeanCorrectedFourier(X:np.array,kernel:np.array):
    kernel1 = kernel - np.mean(kernel)
    X1 = X - np.mean(X)
    Res = signal.fftconvolve(X1, np.flip(kernel1), mode='valid') #mode='same'
    #pos = np.argmax(Res)
    ind = np.unravel_index(np.argmax(Res, axis=None), Res.shape)
    Value = Res[ind]
    #print(Value," ",ind)
    return Res, ind
 
def TemplateMatchingFourier(X:np.array,kernel:np.array):
    #kernel1 = kernel - np.mean(kernel)
    corr = signal.fftconvolve(X, np.flip(kernel), mode='valid') #mode='same'
    #corr = signal.convolve(X, kernel, mode='valid',method='direct') #mode='same'
    X1 = np.square(X)
    G = np.sum(np.square(kernel))
    kernel1 = np.ones(kernel.shape)
    Res = signal.fftconvolve(X1, kernel1, mode='valid') + G - 2 * corr
    #Res = signal.convolve(X1, kernel1, mode='valid',method='direct') + G - 2 * corr
    ind = np.unravel_index(np.argmin(Res, axis=None), Res.shape)
    Value = Res[ind]
    #print(Value," ",ind)
    return Res, ind

def findSymmetry(X:np.array):
    pad = max(X.shape)
    mirror = np.flip(X,axis=1)
    Frame = np.pad(X,((pad,pad),(pad,pad)),mode='constant')
    res = cv2.matchTemplate(Frame,mirror,cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    x = (min_loc[0]-pad+mirror.shape[1])//2 ##?
    return x 

def gradient_symmetry(image:np.array, n = 2):
    if n<=0: return np.array([0])
    size = image.shape[1]
    result = np.empty(size-2*n)
    for i in range(n,size-n):
         result[i-n] = np.sum(np.square(image[:,i-n:i]-np.flip(image[:,i:i+n],axis=1)))
         #result[i] = np.sum(image[i*n:i*n+n,:]-np.flip(image[i*n+n:i*n+2*n,:],axis=0))
    return result

def find_symmetry_simple(X:np.array):
    eyeline = [[-1,-1,-1,-1],[-1,-1,-1,-1]]
    faceline = [-1,-1,-1,-1]
    eyesim=[]
    facecrop = []
    (x,y,w,h) = (0,0,X.shape[1],X.shape[0])
    center = (x + w//2, y + h//2)
    #shift = findSymmetry(X)
    shift = np.argmin(gradient_symmetry(X, n = 15)) + 15
    shift = shift - w//2
    center1 = [center[0]+shift,center[1]]
    faceline[0:2] = [center1[0],center1[1]-h//3]
    faceline[2:4] = [center1[0],center1[1]+h//3]
    rightface = X[:,center1[0]:]
    leftface = X[:,:center1[0]]
    #shift1 = findSymmetry(rightface)
    n = min(15,rightface.shape[1]//2-1)
    shift1 = np.argmin(gradient_symmetry(rightface, n)) + n
    #shift2 = findSymmetry(leftface)
    n = min(15,leftface.shape[1]//2-1)
    shift2 = np.argmin(gradient_symmetry(leftface, n)) + n
    eyeline[0] = [center1[0]+shift1,center1[1]-h//4,center1[0]+shift1,center1[1]+h//4]
    eyeline[1] = [shift2,center1[1]-h//4,shift2,center1[1]+h//4]
    #print(shift,shift1,shift2)
    return faceline, eyeline

def find_symmetry_lines_viola_jones(faces,eyes,X):
    eyeline = [-1,-1,-1,-1]
    faceline = [-1,-1,-1,-1]
    eyesim=[]
    facecrop = []
    if faces:
        (x,y,w,h) = faces[0]
        center = (x + w//2, y + h//2)
        if len(eyes)>=2:
            #we have two eyes data
            for i in range(2):
                (x2,y2,w2,h2) = eyes[i]
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                eyeline[i*2:i*2+2] = [eye_center[0],eye_center[1]]
            v = np.array([eyeline[2]-eyeline[0],eyeline[3]-eyeline[1]])
            v=v/np.linalg.norm(v)
            v2 = np.array([-v[1],v[0]])
            angle = np.arcsin(v[1])
            facecrop = ndimage.rotate(X[y:y+h,x:x+w],np.degrees(angle),reshape=False)
            shift = findSymmetry(facecrop)
            shift = shift - w//2
            center1 = [center[0]+v[0]*shift,center[1]+v[1]*shift]
            faceline[0:2] = [center1[0]-v2[0]*h//2,center1[1]-v2[1]*h//2]
            faceline[2:4] = [center1[0]+v2[0]*h//2,center1[1]+v2[1]*h//2]
        else:
            facecrop = X[y:y+h,x:x+w]
            shift = findSymmetry(facecrop)
            shift = shift - w//2
            center1 = [center[0]+shift,center[1]]
            faceline[0:2] = [center1[0],center1[1]-h//2]
            faceline[2:4] = [center1[0],center1[1]+h//2]
            #we don't have eyes data
            pass

        #self.window.nametowidget("leftframe.originalimage").coords(6,faceline[0],faceline[1],faceline[2],faceline[3])
    else: 
        pass#self.window.nametowidget("leftframe.originalimage").coords(6,0,0,0,0)
    eyeline = [[],[]]
    if faces and eyes:
        if len(eyes)>=2:
            #v,v2,angle are avalible
            for i in range(2):
                (x2,y2,w2,h2) = eyes[i]
                center = (x + x2 + w2//2, y + y2 + h2//2)
                facecrop = ndimage.rotate(X[y+y2:y+y2+h2,x+x2:x+x2+w2],np.degrees(angle),reshape=False)
                shift = findSymmetry(facecrop)
                shift = shift - w2//2
                center1 = [center[0]+v[0]*shift,center[1]+v[1]*shift]
                tmp = [center1[0]-v2[0]*h2//2,center1[1]-v2[1]*h2//2,center1[0]+v2[0]*h2//2,center1[1]+v2[1]*h2//2]
                eyeline[i] = tmp.copy()
                #self.window.nametowidget("leftframe.originalimage").coords(7+i,tmp[0],tmp[1],tmp[2],tmp[3])
        else:
            (x2,y2,w2,h2) = eyes[0]
            center = (x + x2 + w2//2, y + y2 + h2//2)
            facecrop = X[y+y2:y+y2+h2,x+x2:x+x2+w2]
            shift = findSymmetry(facecrop)
            shift = shift - w2//2
            center1 = [center[0]+shift,center[1]]
            tmp = [center1[0],center1[1]-h2//2,center1[0],center1[1]+h2//2]
            eyeline[0] = tmp.copy()
            #self.window.nametowidget("leftframe.originalimage").coords(7,tmp[0],tmp[1],tmp[2],tmp[3])
            #self.window.nametowidget("leftframe.originalimage").coords(8,0,0,0,0)
    else:
        pass#self.window.nametowidget("leftframe.originalimage").coords(7,0,0,0,0)
        #self.window.nametowidget("leftframe.originalimage").coords(8,0,0,0,0)
    return faceline, eyeline