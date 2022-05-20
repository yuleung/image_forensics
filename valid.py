import tensorflow as tf
import cv2
import numpy as np
from skimage import feature
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from networks import DNNs, DNNs_arg_scope
import time
import os
slim = tf.contrib.slim
os.environ['CUDA_VISIBLE_DEVICES']='0'

ColorSpaceNum = 2

#State the dataset directory where the validation set is found
dataset_dir = './dataset/'

#State your checkpoint directory where you can retrieve your model
log_dir = './checkpoint/'

#State your log directory
txt_dir = './log/valid_accuracy.txt'

#State the batch_size to evaluate each time, which can be a lot more than the training batch
batch_size = 100
file_prefix = 'test'
#State the number of epochs to evaluate
num_epochs = 1
keep_prob_set = 1
valid_Gan_num = 2000
valid_Tp_num = 1000
valid_Au_num = 3500


def Sobel(img, Ksize = 3):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0,ksize = Ksize)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize = Ksize)
    absX = cv2.convertScaleAbs(x)  
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    dst = np.clip(dst,0,223)
    return dst


    
def GetMultichannel(imgRGB,threshold):
    imgYCrCb = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2YCrCb)
    imgCr = Scharr(imgYCrCb[:,:,1], threshold)
    imgCb = Scharr(imgYCrCb[:,:,2], threshold)
    #imgCr = Sobel(imgYCrCb[:,:,1], threshold)
    #imgCb = Sobel(imgYCrCb[:,:,2], threshold)
    GLCMofCr = feature.texture.greycomatrix(imgCr, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=threshold+1)[:, :, 0, :]
    GLCMofCb = feature.texture.greycomatrix(imgCb, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=threshold+1)[:, :, 0, :]
    IMG = np.concatenate((GLCMofCr,GLCMofCb),axis=2)
    return IMG


def _parse_function(example_proto):   
    features = tf.parse_single_example(example_proto,
                               features={
                                    'imaString': tf.FixedLenFeature([], tf.string),
                                    'label': tf.FixedLenFeature([], tf.int64)
                                   })
    image = tf.decode_raw(features['imaString'],tf.uint8)
    image = tf.reshape(image, [500,500,3])
    label = features['label']
    #image = GetMultichannel(image)
    return image, label

def GetTheNumOfImgInAEpoch(file_prefix):    
    # Count the total number of examples in dataset
    num_samples = 0
    file_pattern_for_counting =  file_prefix
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1
    return num_samples

def get_imges_labels(sess):
    IMGS = np.zeros((batch_size, 224, 224, 4 * ColorSpaceNum), dtype=np.float32)
    imgs, Labels = sess.run([next_imgs, next_labels]) 
    for j in range(batch_size):  
        img = imgs[j].astype(np.uint8)
        IMG = GetMultichannel(img)
        IMGS[j] = IMG.astype(np.float32)
    return IMGS, Labels



tf.logging.set_verbosity(tf.logging.INFO)
#Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
file_list = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_prefix)]
num_samples = GetTheNumOfImgInAEpoch(file_prefix)
print(num_samples)


IMGS = tf.placeholder(tf.float32, (batch_size, 224, 224, 4 * ColorSpaceNum))
Labels = tf.placeholder(tf.int32, (batch_size,))
keep_prob = tf.placeholder(tf.float32)
#Create some information about the training steps
num_batches_per_epoch = num_samples / batch_size

#Now create the inference model but set is_training=False
with slim.arg_scope(DNNs_arg_scope()):
    logits, end_points = DNNs(IMGS, num_classes = 3,keep_prob = keep_prob, is_training = True)

probabilities = end_points['Predictions']
predictions = tf.argmax(probabilities, 1) 

one_hot_labels = slim.one_hot_encoding(Labels, 3)   
cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)#
total_loss = tf.losses.get_total_loss() 

#caculate confusion_matrix
all_labels = tf.placeholder(tf.int32,(num_samples))
all_predictions = tf.placeholder(tf.float32,(num_samples))
confusion_matrix = tf.confusion_matrix(all_labels, all_predictions,num_classes = 3)

global_step = get_or_create_global_step()

variables_to_restore = slim.get_variables_to_restore()
saver = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    sess.run(tf.initialize_local_variables())

    checkpoint_dir = os.path.join(log_dir, 'checkpoint')
    checkpoint_file_path_list = []
    #get all checkpoint_file list
    for line in open(checkpoint_dir):
        line_path = line.split(':')[-1][:-1]
        checkpoint_file_path_list.append(line_path.split('"')[1])
    checkpoint_file_path_list.pop(0)
    
    max_F1_macro_score = 0
    Tmp_Recall_Gan = 0
    Tmp_Recall_Tp = 0
    Tmp_Recall_Au = 0
    Tmp_Precision_Gan = 0
    Tmp_Precision_Tp = 0
    Tmp_Precision_Au = 0
    Tmp_macro_Precision = 0
    Tmp_macro_Recall = 0
    Tmp_confusion_matrix_result = 0
    Tmp_global_step_count = 0
    
    #valid evey checkpoint_file successively
    for checkpoint_file_path in checkpoint_file_path_list:
        print(checkpoint_file_path)
        saver.restore(sess, checkpoint_file_path)
        dataset = tf.data.TFRecordDataset(file_list)  
        dataset = dataset.map(_parse_function)  
        dataset = dataset.batch(batch_size)  
        iterator = dataset.make_one_shot_iterator()  
        next_imgs, next_labels = iterator.get_next()  
        
        valid_labels = np.array([],dtype=int)
        valid_predicteds =  np.array([],dtype=float)
        correct_num = 0
        for step in range(int(num_batches_per_epoch * num_epochs)):   
            imgs,labels = get_imges_labels(sess)
            loss,img_predictions, global_step_count = sess.run([total_loss, predictions,global_step], feed_dict= {IMGS:imgs, Labels:labels,keep_prob:keep_prob_set})
            valid_labels = np.concatenate((valid_labels, labels))
            print(labels)
            print(img_predictions)
            valid_predicteds = np.concatenate((valid_predicteds, img_predictions))
            
            
        #caculate F1_macro_score accroding the confusion_matrix
        confusion_matrix_result = sess.run(confusion_matrix, feed_dict={all_labels:valid_labels, all_predictions:valid_predicteds})
        Recall_Gan = confusion_matrix_result[2][2] / np.sum(confusion_matrix_result[2])
        Recall_Tp = confusion_matrix_result[1][1] / np.sum(confusion_matrix_result[1])
        Recall_Au = confusion_matrix_result[0][0] / np.sum(confusion_matrix_result[0])
        print(confusion_matrix_result)
        Precision_Gan = confusion_matrix_result[2][2] / (confusion_matrix_result[2][2] + confusion_matrix_result[0][2] * valid_Gan_num / valid_Au_num + confusion_matrix_result[1][2] * valid_Gan_num / valid_Tp_num)
        print(Precision_Gan)
        Precision_Tp =  confusion_matrix_result[1][1] / (confusion_matrix_result[1][1] + confusion_matrix_result[0][1] * valid_Tp_num / valid_Au_num + confusion_matrix_result[2][1] * valid_Tp_num / valid_Gan_num)
        print(Precision_Tp)
        Precision_Au = confusion_matrix_result[0][0] /(confusion_matrix_result[0][0] + confusion_matrix_result[1][0] * valid_Au_num / valid_Tp_num + confusion_matrix_result[2][0] * valid_Au_num / valid_Gan_num)
        print(Precision_Au)
        macro_Precision = (Precision_Gan + Precision_Tp + Precision_Au) / 3
        macro_Recall = (Recall_Gan + Recall_Tp + Recall_Au) / 3
        F1_macro_score = 2 * macro_Precision * macro_Recall / (macro_Precision + macro_Recall)
        
        if max_F1_macro_score < F1_macro_score:
            max_F1_macro_score = F1_macro_score
            Tmp_Recall_Gan = Recall_Gan
            Tmp_Recall_Tp = Recall_Tp
            Tmp_Recall_Au = Recall_Au
            Tmp_Precision_Gan = Precision_Gan
            Tmp_Precision_Tp = Precision_Tp
            Tmp_Precision_Au = Precision_Au
            Tmp_macro_Precision = macro_Precision
            Tmp_macro_Recall = macro_Recall
            Tmp_confusion_matrix_result = confusion_matrix_result
            Tmp_global_step_count = global_step_count
        #save the result at this checkpoint file
        with open(txt_dir, 'a+') as File:
            line_str = (str(confusion_matrix_result[0][0]) + '  ' + str(confusion_matrix_result[0][1]) + '  ' + str(confusion_matrix_result[0][2]) + '\n'
            + str(confusion_matrix_result[1][0]) + '  ' + str(confusion_matrix_result[1][1]) + '  ' + str(confusion_matrix_result[1][2]) + '\n'
            + str(confusion_matrix_result[2][0]) + '  ' + str(confusion_matrix_result[2][1]) + '  ' + str(confusion_matrix_result[2][2]) + '\n' 
            + 'Train: '  +'loss: ' + str(loss)+'  global_step: ' + str(global_step_count) + '  Recall_Tp: ' + str(Recall_Tp) + '  Recall_Gan: ' + str(Recall_Gan) + '  Recall_Au: ' + str(Recall_Au) + '  macro_Recall: '+ str(macro_Recall) +'  F1_macro_score: '+ str(F1_macro_score) + '\n' + '\n')
            File.write(line_str)
            
    with open(txt_dir, 'a+') as File:
        File.write('Best Result: ' + '\n')
        line_str = (str(Tmp_confusion_matrix_result[0][0]) + '  ' + str(Tmp_confusion_matrix_result[0][1]) + '  ' + str(Tmp_confusion_matrix_result[0][2]) + '\n'
        + str(Tmp_confusion_matrix_result[1][0]) + '  ' + str(Tmp_confusion_matrix_result[1][1]) + '  ' + str(Tmp_confusion_matrix_result[1][2]) + '\n'
        + str(Tmp_confusion_matrix_result[2][0]) + '  ' + str(Tmp_confusion_matrix_result[2][1]) + '  ' + str(Tmp_confusion_matrix_result[2][2]) + '\n' 
        + '  global_step: ' + str(Tmp_global_step_count) + '  Recall_Tp: ' + str(Tmp_Recall_Tp) + '  Recall_Gan: ' + str(Tmp_Recall_Gan) + '  Recall_Au: ' + str(Tmp_Recall_Au) + '  macro_Recall: '+ str(Tmp_macro_Recall) +'  F1_macro_score: '+ str(Tmp_F1_macro_score) + '\n' + '\n')
        File.write(line_str)        
