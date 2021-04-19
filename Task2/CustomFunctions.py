import numpy as np
import cv2 as cv2
from scipy import signal

def normalize(X:np.array):
    a = np.min(X)
    b = np.max(X)
    return 256*(X-a)/(b-a)

def SSD(X:np.array,Y:np.array):
    return np.sum(np.square(X-Y))

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