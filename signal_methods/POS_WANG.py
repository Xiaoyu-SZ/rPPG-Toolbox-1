import cv2
import numpy as np
import math
from scipy import signal
from scipy import sparse
from scipy import linalg
from scipy import io as scio
from skimage.util import img_as_float
from signal_methods import utils

def POS_WANG(VideoFile,ECGFile,PPGFile,PlotTF):
    SkinSegmentTF = False
    LPF = 0.7
    HPF = 2.5
    WinSec = 1.6
    T, RGB,FS= process_video(VideoFile)
    #lines and comments
    N = RGB.shape[0]
    H = np.zeros((1,N))
    l = math.ceil(WinSec*FS)
    C = np.zeros((1,3))
    for n in range(N):
        m = n-l
        if(m>=0):
            Cn = np.true_divide(RGB[m:n,:],np.mean(RGB[m:n,:],axis=0))
            Cn = np.mat(Cn).H
            S = np.matmul(np.array([[0,1,-1],[-2,1,1]]),Cn)
            h = S[0,:]+(np.std(S[0,:])/np.std(S[1,:]))*S[1,:]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0,temp] = h[0,temp]-mean_h
            H[0,m:n] = H[0,m:n]+(h[0])

    BVP = H
    BVP = utils.detrend(np.mat(BVP).H,100)
    BVP = np.asarray(np.transpose(BVP))[0]
    B,A = signal.butter(1,[0.75/FS*2,3/FS*2],'bandpass')
    BVP = signal.filtfilt(B,A,BVP.astype(np.double))
    PR = utils.prpsd(BVP,FS,40,240,True)
    return BVP,PR

def process_video(VideoFile):
    #Standard:
    VidObj = cv2.VideoCapture(VideoFile)
    FrameRate = VidObj.get(cv2.CAP_PROP_FPS)
    T = []
    RGB = []
    success, frame = VidObj.read()
    CurrentTime = VidObj.get(cv2.CAP_PROP_POS_MSEC)

    while(success):
        T.append(CurrentTime)
        #TODO: if different region
        frame = cv2.cvtColor(np.array(frame).astype('float32'), cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        sum = np.sum(np.sum(frame,axis=0),axis=0)
        RGB.append(sum/(frame.shape[0]*frame.shape[1]))
        success, frame = VidObj.read()
        CurrentTime = VidObj.get(cv2.CAP_PROP_POS_MSEC)
    return np.asarray(T),np.asarray(RGB),FrameRate



# DataDirectory           = 'test_data\\'
# VideoFile               = DataDirectory+ 'video_example3.avi'#TODO:deal with files not found error
# FS                      = 120
# StartTime               = 0
# Duration                = 60
# ECGFile                 = DataDirectory+ 'ECGData.mat'
# PPGFile                 = DataDirectory+ 'PPGData.mat'
# PlotTF                  = False
#
# POS_WANG(VideoFile,FS,StartTime,Duration,ECGFile,PPGFile,PlotTF)
