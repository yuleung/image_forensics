import tensorflow as tf
import numpy as np
import cv2
import os
import random
from skimage import feature
from tensorflow.python.ops import control_flow_ops
slim = tf.contrib.slim
from networks import DNNs, DNNs_arg_scope
import time
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
os.environ['CUDA_VISIBLE_DEVICES']='1'


#if more than one tfrecord file you have generated, you can shuffle them to get higher randomness
def getNumEpochTfrecordWithShuffle(Path, Epoch):   
    tfrecords = os.listdir(Path)
    for i in range(len(tfrecords)):
        tfrecords[i] = os.path.join(Path, tfrecords[i])
    shuffleNumEpochTfrecordsList = []
    for i in range(Epoch):
        random.shuffle(tfrecords)
        shuffleNumEpochTfrecordsList.append(tfrecords)
    return shuffleNumEpochTfrecordsList
    
# Parse tfrecord file to rebuild images and labels
def _parse_function(example_proto, threshold):   
    features = tf.parse_single_example(example_proto,
                               features={
                                    'imaString': tf.FixedLenFeature([], tf.string),
                                    'label': tf.FixedLenFeature([], tf.int64)
                                   })
    image = tf.decode_raw(features['imaString'],tf.float32)
    image = tf.reshape(image, [threshold, threshold,8])
    label = features['label']
    return image, label

#to get how many samples in A Epoch
def getTheNumOfImgInAEpoch(trainFiles): 
    num_samples = 0
    filePatternForCounting =  trainFiles
    tfrecords_to_count = [os.path.join(tfrecordDir, file) for file in os.listdir(tfrecordDir) if file.startswith(filePatternForCounting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1
    return num_samples




if __name__ == '__main__':
    # The path of tfrecord file          ##You should change it to your own path
    tfrecordDir = '/home/manager/LY/Dataset/tfrecord/train_scharr_192'  
    # The prefix of file 
    trainFile = 'train'
    # The path of checkpoint file   ##You should change it to your own path
    checkpointDir = '/home/manager/LY/train_test/checkpoint_scharr_192'
    # The path of result file       ##You should change it to your own path
    logDir = '/home/manager/LY/train_test/log_scharr_192/step_accuracy.txt'
    bacthSize = 56
    threshold = 192
    howManyTimeShuffleFile = 1
    howManyRepeatFileList = 80
    initialLearningRate = 0.0005
    learningRateDecayFactor = 0.85
    keep_prob_set = 0.5
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    fileList = getNumEpochTfrecordWithShuffle(tfrecordDir, howManyTimeShuffleFile)   

    # Read tfrecord file
    dataset = tf.data.TFRecordDataset(fileList)                 
    dataset = dataset.map(_parse_function(threshold))                       
    dataset = dataset.repeat(howManyRepeatFileList)                 
    dataset = dataset.shuffle(buffer_size = 3000)                   
    dataset = dataset.batch(bacthSize)                                     
    iterator = dataset.make_one_shot_iterator()
    nextImgs, nextLabels = iterator.get_next() 


    def getImgesLabels(sess):
        imgs, labels = sess.run([nextImgs, nextLabels]) 
        return imgs, labels
        
    # Get how many batches and samples in a epoch
    numSamplesPerEpoch = getTheNumOfImgInAEpoch(trainFile)    
    numBatchesPerEpoch = numSamplesPerEpoch //  bacthSize   
    decay_steps = 600
    

    # Set the verbosity to INFO level
    tf.logging.set_verbosity(tf.logging.INFO)  

    IMGS = tf.placeholder(tf.float32, (bacthSize, threshold, threshold, 8))
    Labels = tf.placeholder(tf.int32, (bacthSize,))
    keep_prob = tf.placeholder(tf.float32)
    
    # Set default configuration
    with slim.arg_scope(DNNs_arg_scope()):  
        logits, end_points = DNNs(IMGS, num_classes=3, keep_prob = keep_prob, is_training = True)
        
    # Labels to one-hot encoding
    one_hot_labels = slim.one_hot_encoding(Labels, 3)   


    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)#
    totalLoss = tf.losses.get_total_loss() 
    globalStep = get_or_create_global_step()   

    # decayed_learning_rate = learining_rate * decay_rate ^ ( global_step/decay_steps )
    lr = tf.train.exponential_decay(    
                learning_rate = initialLearningRate, 
                global_step = globalStep,    
                decay_steps = decay_steps,
                decay_rate = learningRateDecayFactor,
                staircase = True) 

    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    # Create the trainOp.
    trainOp = slim.learning.create_train_op(totalLoss, optimizer) 
    predictions = tf.argmax(end_points['Predictions'], 1)    
    
    # The probabilities of the samples in a batch
    probabilities = end_points['Predictions']  
    accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, Labels)
    metrics_op = tf.group(accuracy_update, probabilities)  

    def train_step(sess, trainOp, globalStep,imgs,labels):
        start_time = time.time()
        totalLoss, globalStepCount, _ = sess.run([trainOp, globalStep,metrics_op],feed_dict={IMGS:imgs, Labels:labels,keep_prob:keep_prob_set })
        time_elapsed = time.time() - start_time
        
        # Run the logging to print some results
        logging.info('global step %s: loss: %.4f (%.2f sec/step)', globalStepCount, totalLoss, time_elapsed)
        #return total loss and which step 
        return totalLoss, globalStepCount    

    if not os.path.exists(checkpointDir):
        os.mkdir(checkpointDir)
        
    saver = tf.train.Saver(max_to_keep=200)
    ckptPath = os.path.join(checkpointDir, 'mode.ckpt')
    lossAvg = 0
    lossTotal = 0
    with tf.Session(config=config) as sess:
    
        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())
        
        if os.listdir(checkpointDir):
            model_file = tf.train.latest_checkpoint(checkpointDir)
            saver.restore(sess, model_file)
            print('total samples: %d' %(numBatchesPerEpoch * howManyTimeShuffleFile * howManyRepeatFileList))
            
        for step in range(numBatchesPerEpoch * howManyTimeShuffleFile * howManyRepeatFileList): 
            imgs, labels = getImgesLabels(sess)
            #for each epoch
            if step % numBatchesPerEpoch == 0:  
                logging.info('Epoch %s/%s', step / numBatchesPerEpoch + 1, howManyTimeShuffleFile * howManyRepeatFileList)
                learning_rate_value, accuracy_value = sess.run([lr, accuracy],feed_dict={IMGS:imgs, Labels:labels,keep_prob:keep_prob_set })
                logging.info('Current Learning Rate: %s', learning_rate_value)
                logging.info('Current Streaming Accuracy: %s', accuracy_value)
                logits_value, probabilities_value, predictions_value = sess.run([logits, probabilities, predictions],feed_dict={IMGS:imgs, Labels:labels,keep_prob:keep_prob_set })
                print('logits: \n', logits_value[:5])
                print('Probabilities: \n', probabilities_value[:5])
                print('lables:    :{}\n'.format(labels))
                print('predictions:{}\n'.format(predictions_value))
                print (lossTotal)
                lossTotal = 0
   
            loss, globalStepCount = train_step(sess, trainOp, globalStep,imgs,labels)
            lossAvg += loss
            lossTotal += loss
            
            #how many step to save a ckpt file
            if step % 150 == 0:    
                learning_rate_value, accuracy_value = sess.run([lr, accuracy],feed_dict={IMGS:imgs, Labels:labels})
                print ('learning_rate_value: {}\n accuracy_value: {}'.format(learning_rate_value,accuracy_value))
                with open(logDir, 'a+') as File:
                    line_str = 'learining_rate: '+str(learning_rate_value) + '  global_step: ' + str(globalStepCount) + '  loss: '+ str(lossAvg/150) +'  accuracy_value: ' + str(accuracy_value) + '\n'
                    print(line_str)
                    File.write(line_str)
                lossAvg = 0 
                saver.save(sess, ckptPath, globalStep = globalStep, write_meta_graph=False)
                if not os.path.exists('/home/manager/LY/train_test/checkpoint_scharr_192/*.meta'):
                    saver.export_meta_graph('/home/manager/LY/train_test/checkpoint_scharr_192/mode.ckpt.meta')

