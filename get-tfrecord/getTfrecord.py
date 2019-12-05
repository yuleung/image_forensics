#get-tfrecord file
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
    

def Sobel(img):
    x = cv2.Scharr(img, cv2.CV_16S, 1, 0)
    y = cv2.Scharr(img, cv2.CV_16S, 0, 1)
    dst = cv2.addWeighted(abs(x), 0.5, abs(y), 0.5, 0)
    dst = np.clip(dst,0,191)
    return dst

def GetMultichannel(imgRGB):
    imgYCrCb = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2YCrCb)
    imgYCrCb1 = Sobel(imgYCrCb[:,:,1])
    imgYCrCb2 = Sobel(imgYCrCb[:,:,2])
    imgYCrCb1 = feature.texture.greycomatrix(imgYCrCb1, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=192)[:,:,0,:]
    imgYCrCb2 = feature.texture.greycomatrix(imgYCrCb2, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=192)[:, :, 0, :]
    IMG = np.concatenate((imgYCrCb1,imgYCrCb2),axis=2)
    return IMG

def GetLable(FileName):    #AU:0 TP:1 GAN:2
    Split = FileName.split('_')
    if Split[0] == 'BGAN':
        return 2
    elif Split[0] == 'Sp' or Split[0] == 'Tp':
        return 1
    else:
        return 0

'''
def GetLable(FileName):    #AU:0 TP:1 GAN:2
    if len(FileName)>25:
        return 0
    else:
        return 1
      '''
#获得数据目录
#PathPreTrain = r'/home/manager/SeXceptionResnet/Dataset/NEW/train_casia1-2_4844'
#PathRecord = r'/home/manager/SeXceptionResnet/Dataset/NEW_tfrecord/train_casia1-2_4844'
#PathPreTrain = r'/home/manager/SeXceptionResnet/Dataset/NEW/test_casia1-2_1200'
#PathRecord = r'/home/manager/SeXceptionResnet/Dataset/NEW_tfrecord/test_casia1-2_1200'

#PathPreTrain = r'/home/manager/SeXceptionResnet/Dataset/NEW/train_styleGan'
#PathRecord = r'/home/manager/SeXceptionResnet/Dataset/NEW_tfrecord/train_scharr_256_stylegan_10000'
#PathPreTrain = r'/home/manager/SeXceptionResnet/Dataset/NEW/test_styleGan'
#PathRecord = r'/home/manager/SeXceptionResnet/Dataset/NEW_tfrecord/test_scharr_256_stylegan_3000'
#PathPreTrain = r'/home/manager/SeXceptionResnet/Dataset/NEW/train_4123'
#PathRecord = r'/home/manager/SeXceptionResnet/Dataset/NEW_tfrecord/train_scharr_160_4123'
PathPreTrain = r'/home/manager/SeXceptionResnet/Dataset/NEW/test_1000'
PathRecord = r'/home/manager/SeXceptionResnet/Dataset/NEW_tfrecord/test_scharr_160_1000'
#PathPreTrain = r'/home/manager/SeXceptionResnet/Dataset/NEW/test_1000'
#PathRecord = r'/home/manager/SeXceptionResnet/Dataset/NEW_tfrecord/test_scharr_288_1000'

#PathPreTrain = r'/home/manager/SeXceptionResnet/Dataset/NEW/train_casia1-2_4844'
#PathRecord = r'/home/manager/SeXceptionResnet/Dataset/NEW_tfrecord/train_casia1-2_4844'
#PathPreTrain = r'/home/manager/SeXceptionResnet/Dataset/NEW/test_casia1-2_1200'
#PathRecord = r'/home/manager/SeXceptionResnet/Dataset/NEW_tfrecord/test_casia1-2_1200'
##获得目录下的文件名存为list
FileOfPreTrain = os.listdir(PathPreTrain)

#打乱List中数据
random.shuffle(FileOfPreTrain)

#获得文件并生成多通道图像和Lable --->Tfrecord  Test
imgNumber = 0
RecordFileNum = 0
TfrecordsFilename = ('train_%.3d_.tfrecord' % RecordFileNum)
Path = os.path.join(PathRecord, TfrecordsFilename)
writer = tf.python_io.TFRecordWriter(Path)
for FileName in FileOfPreTrain:
    FilePath = os.path.join(PathPreTrain, FileName)
    img = cv2.imread(FilePath)   #.shape(heigth,width)
    img = GetMultichannel(img)
    img = img.astype(np.float32)
    lable = GetLable(FileName)
    imgString = img.tobytes()
    example = tf.train.Example(features = tf.train.Features(feature ={
                       'imaString': _bytes_features(imgString),
                        'label': _int64_features(lable)
                        }))
    if imgNumber % 100 == 0:
        print(imgNumber)
    writer.write(example.SerializeToString())
    imgNumber += 1
print('Done the %d Tfrecord of test'%RecordFileNum)
writer.close()
