import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
slim = tf.contrib.slim

def DNNs(inputs,
             keep_prob,
             num_classes=3,
             is_training=True,
             scope='DNNs'):

    with tf.variable_scope('dnns') as sc:
        end_points_collection = sc.name + '_end_points'

        with slim.arg_scope([slim.separable_conv2d], depth_multiplier=1), \
             slim.arg_scope([slim.separable_conv2d, slim.conv2d, slim.avg_pool2d],
                            outputs_collections=[end_points_collection]), \
             slim.arg_scope([slim.batch_norm], is_training=is_training):
              
            #input： T * T * 8  
            net = slim.conv2d(inputs, 64, [5, 5], stride=4, scope='block1_conv1')    
            net = slim.batch_norm(net, scope='block1_bn1')
            net = tf.nn.relu(net, name='block1_relu1') 
            net = slim.conv2d(net, 128, [3, 3],scope='block1_conv2')  
            net = slim.batch_norm(net, scope='block1_bn2')
            net = tf.nn.relu(net, name='block1_relu2')
            net = slim.conv2d(net, 256, [3, 3],scope='block1_dws_conv1') 
            net = slim.batch_norm(net, scope='block1_bn3')
            net = tf.nn.relu(net, name='block1_relu3')
            residual = slim.conv2d(net, 256, [1, 1], stride=2, scope='block1_res_conv')   
            residual = slim.batch_norm(residual, scope='block1_res_bn')

            # Block 2
            net = slim.separable_conv2d(net, 256, [3, 3], scope='block2_dws_conv1') 
            net = slim.batch_norm(net, scope='block2_bn1')
            net = tf.nn.relu(net, name='block2_relu1')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block2_max_pool')   
            net = slim.batch_norm(net, scope='block2_bn3')
            net = tf.add(net, residual, name='block2_add')      
            net = tf.nn.relu(net, name='block2_relu3')
            residual = slim.conv2d(net, 512, [1, 1], stride=2, scope='block2_res_conv')  
            residual = slim.batch_norm(residual, scope='block2_res_bn')

            # Block 3
            net = slim.separable_conv2d(net, 512, [3, 3], scope='block3_dws_conv1')      
            net = slim.batch_norm(net, scope='block3_bn1')
            net = tf.nn.relu(net, name='block3_relu1')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block3_max_pool')  
            net = slim.batch_norm(net, scope='block3_bn3')
            net = tf.add(net, residual, name='block3_add')             
            net = tf.nn.relu(net, name='block3_relu3')
            residual = slim.conv2d(net, 1024, [1, 1], stride=1, scope='block3_res_conv')    
            residual = slim.batch_norm(residual, scope='block3_res_bn')
            
            # Block 4
            net = slim.separable_conv2d(net, 1024, [3, 3], scope='block4_dws_conv1')   
            net = slim.batch_norm(net, scope='block4_bn1')
            net = tf.nn.relu(net, name='block4_relu1')
            net = slim.separable_conv2d(net, 1024, [3, 3], scope='block4_dws_conv2')     
            net = slim.batch_norm(net, scope='block4_bn2')
            net = tf.nn.relu(net, name='block4_relu2')
            net = slim.separable_conv2d(net, 1024, [3, 3], scope='block4_dws_conv3')     
            net = slim.batch_norm(net, scope='block4_bn3')
            net = tf.add(net, residual, name='block4_add')               
            net = tf.nn.relu(net, name='block4_relu3')
            residual = slim.conv2d(net, 1024, [1, 1], stride=1, scope='block4_res_conv')    
            residual = slim.batch_norm(residual, scope='block4_res_bn')
            
            # Block 5
            net = slim.separable_conv2d(net, 1024, [3, 3], scope='block5_dws_conv1')  
            net = slim.batch_norm(net, scope='block5_bn1')
            net = tf.nn.relu(net, name='block5_relu1')
            net = slim.separable_conv2d(net, 1024, [3, 3], scope='block5_dws_conv2')  
            net = slim.batch_norm(net, scope='block5_bn2')
            net = tf.nn.relu(net, name='block5_relu2')
            net = slim.separable_conv2d(net, 1024, [3, 3], scope='block5_dws_conv3') 
            net = slim.batch_norm(net, scope='block5_bn3')
            net = tf.add(net, residual, name='block5_add')                         
            net = tf.nn.relu(net, name='block5_relu3')


            net = slim.separable_conv2d(net, 2048, [3, 3], scope='block6_dsp_conv1')   
            net = slim.batch_norm(net, scope='block6_bn1')
            net = tf.nn.relu(net, name='block6_relu2')
            net = slim.separable_conv2d(net, 2048, [3, 3], scope='block6_dsp_conv2')    
            net = slim.batch_norm(net, scope='block6_bn2')
            net = tf.nn.relu(net, name='block6_relu2')

            fc1 = global_avg_pool(net, name = 'Global_avg_pooling')
            dropunit1 = tf.nn.dropout(fc1,keep_prob = keep_prob)
            fc2 = tf.layers.dense(dropunit1, units = 2048,activation = tf.nn.relu)
            logits = tf.layers.dense(fc2,units=num_classes,activation=None)
            predictions = slim.softmax(logits,scope = 'predictions')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            end_points['Logits'] = logits  
            end_points['Predictions'] = predictions

        return logits, end_points


def DNNs_arg_scope(weight_decay=0.0001,
                       batch_norm_decay=0.95,
                       batch_norm_epsilon=0.001):

    # Set weight_decay for weights in conv2d and separable_conv2d layers.
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=None,
                        activation_fn=None):
        # Set parameters for batch_norm.
        with slim.arg_scope([slim.batch_norm],
                            decay=batch_norm_decay,
                            epsilon=batch_norm_epsilon) as scope:
            return scope
            
            