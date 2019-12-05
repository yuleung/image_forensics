import tensorflow as tf
import numpy as np
import os
import random
import cv2
from skimage import feature

def _bytes_features(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _int64_features(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))
    

def Scharr(img, threshold):
    x = cv2.Scharr(img, cv2.CV_16S, 1, 0)
    y = cv2.Scharr(img, cv2.CV_16S, 0, 1)
    dst = cv2.addWeighted(abs(x), 0.5, abs(y), 0.5, 0)
    dst = np.clip(dst,0,threshold)
    return dst
    
'''
def Sobel(img, Ksize = 3):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0,ksize = Ksize)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize = Ksize)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    dst = np.clip(dst,0,threshold)
    return dst
'''


def getGLCM(imgRGB,threshold):
    imgYCrCb = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2YCrCb)
    imgCr = Scharr(imgYCrCb[:,:,1], threshold)
    imgCb = Scharr(imgYCrCb[:,:,2], threshold)
    #imgCr = Sobel(imgYCrCb[:,:,1], threshold)
    #imgCb = Sobel(imgYCrCb[:,:,2], threshold)
    GLCMofCr = feature.texture.greycomatrix(imgCr, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=threshold+1)[:, :, 0, :]
    GLCMofCb = feature.texture.greycomatrix(imgCb, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=threshold+1)[:, :, 0, :]
    IMG = np.concatenate((GLCMofCr,GLCMofCb),axis=2)
    return IMG

def getLable(FileName):    #AU:0 TP:1 GAN:2
    Split = FileName.split('_')
    if Split[0] == 'BGAN':
        return 2
    elif Split[0] == 'Sp' or Split[0] == 'Tp':
        return 1
    else:
        return 0


def getTfrecord(pathOfFile, pathOfSaveRecord, threshold):
    Files = os.listdir(pathOfFile)
    #shuffle the data
    random.shuffle(Files)
    #to count how many img have been processed
    imgNumber = 0
    #when the tfrecord file is too large, you can set a looping conditions for sharding
    RecordFileNum = 0
    TfrecordsFilename = ('train_%.3d_.tfrecord' % RecordFileNum)
    Path = os.path.join(pathOfSaveRecord, TfrecordsFilename)
    writer = tf.python_io.TFRecordWriter(Path)
    for fileName in Files:
        FilePath = os.path.join(pathOfFile, fileName)
        img = cv2.imread(FilePath)  
        img = getGLCM(img,threshold)
        img = img.astype(np.float32)
        lable = getLable(fileName)
        imgString = img.tobytes()
        example = tf.train.Example(features = tf.train.Features(feature ={
                           'imaString': _bytes_features(imgString),
                            'label': _int64_features(lable)
                            }))
        if imgNumber % 100 == 0:
            print('%d pictures have been processed' %imgNumber)
        writer.write(example.SerializeToString())
        imgNumber += 1
    writer.close()
    
    
if __name__ == '__main__':
    #the path of the train images, you should change it to your path
    pathOfTrainFile = r'/home/manager/LY/Dataset/train'   
    #the path to save the tfrecord file of train, you should change it to your path
    pathOfTrainRecord = r'/home/manager/LY/Dataset/train_scharr_192'     
    #the path of the test images, you should change it to your path
    pathOfTestFile = r'/home/manager/LY/Dataset/test_1000'    
    #the path to save the tfrecord file of train, you should change it to your path
    pathOfTestRecord = r'/home/manager/LY/Dataset/tfrecord/test_scharr_192'     
    #this threshold is described in the paper
    threshold = 191  
    
    getTfrecord(pathOfTrainFile, pathOfTrainRecord, threshold)
    print('Done of the tfrecord of test!')
    getTfrecord(pathOfTestFile, pathOfTestRecord, threshold)
    print('Done of the tfrecord of test!')
    print('Done!')
