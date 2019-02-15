"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os

import numpy as np
import tensorflow as tf

from lilalexnet import LilAlexNet, fc
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator

"""
Configuration Part.
"""

#TODO: Path to the textfiles for the trainings and validation set
train_file = '/path/to/train.txt'
val_file = '/path/to/val.txt'

# Learning params
learning_rate = 0.01
#TODO: original value is 10.
num_epochs = 1
batch_size = 128

# Network params
dropout_rate = 0.5
num_classes = 51
train_layers = ['fc6', 'fc7', 'class']

# How often we want to write the tf.summary data to disk
display_step = 100

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "/tmp/finetune_alexnet/tensorboard"
checkpoint_path = "/tmp/finetune_alexnet/checkpoints"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
validation_init_op = iterator.make_initializer(val_data.data)

# TF placeholder for graph input and output
x_rgb = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
x_depth = tf.placeholder(tf.float32, [batch_size, 227, 227, 1])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
streams = ["RGB", "Depth"]

with tf.variable_scope(streams[0]):
    rgb_stream = LilAlexNet(streams[0], x_rgb, keep_prob, num_classes, train_layers)
    rgb_fc7 = rgb_stream.fc7
    score_rgb = fc(rgb_fc7, 4096, num_classes, relu=False, name='class')

with tf.variable_scope(streams[1]):
    #TODO: jet colorization. Right now I just broadcast the input along the final channel .
    x_depth = tf.broadcast_to(x_depth, [-1, 227, 227, 3])
    depth_stream = LilAlexNet(streams[1], x_depth, keep_prob, num_classes, train_layers)
    d_fc7 = depth_stream.fc7
    score_depth = fc(d_fc7, 4096, num_classes, relu=False, name='class')

# Link variable to model output
_fc_concat = tf.concat([rgb_fc7, d_fc7], axis=1)
with tf.variable_scope("fusion"):
    fc1_fus = fc(_fc_concat, 4096*2, 4096, relu=False)
    score_fusion = fc(fc1_fus, 4096, num_classes, relu=False, name='class_fusion')

# List of trainable variables of the layers we want to train
var_list_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, streams[0])
var_list_1 = [v for v in var_list_1 if v.name.split('/')[1] in train_layers]

var_list_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, streams[1])
var_list_2 = [v for v in var_list_2 if v.name.split('/')[1] in train_layers]

var_list_top_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fusion")

var_list = var_list_1 + var_list_2 + var_list_top_layers

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss_rgb = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score_rgb,
                                                                  labels=y))
    loss_depth = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score_depth,
                                                                  labels=y))
    loss_fusion = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score_fusion,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss_rgb, var_list_1)
    gradients_1 = list(zip(gradients, var_list_1))

    gradients = tf.gradients(loss_depth, var_list_2)
    gradients_2 = list(zip(gradients, var_list_2))

    gradients = tf.gradients(loss_fusion, var_list)
    gradients_3 = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_stream1_op = optimizer.apply_gradients(grads_and_vars=gradients_1)

    train_stream2_op = optimizer.apply_gradients(grads_and_vars=gradients_2)

    train_fusion_op = optimizer.apply_gradients(grads_and_vars=gradients_3)


# Add gradients to summary
gradients = gradients_1 + gradients_2 + gradients_3
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss_fusion)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score_fusion, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))



def load_initial_weights(self, session, bottom_trainable):
    """Load weights from file into network.

    As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
    come as a dict of lists (e.g. weights['conv1'] is a list) and not as
    dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
    'biases') we need a special load function
    """
    # Load the weights into memory
    weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

    # Loop over all layer names stored in the weights dict
    for op_name in weights_dict:

        # Check if layer should be trained from scratch
        if op_name not in self.SKIP_LAYER:

            for stream in streams:
                with tf.variable_scope(stream+"/"+op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', bottom_trainable)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', bottom_trainable)
                            session.run(var.assign(data))



# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    load_initial_weights(sess, bottom_trainable=False)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        #mix the three steps in a loop.
        for _train_op in [train_stream1_op, train_stream2_op, train_fusion_op]:
            for step in range(train_batches_per_epoch):

                # get next batch of data
                rgb_img_batch, depth_img_batch , label_batch = sess.run(next_batch)

                # And run the training op
                sess.run(_train_op, feed_dict={x_rgb: rgb_img_batch,
                                               x_depth: depth_img_batch,
                                               y: label_batch,
                                               keep_prob: dropout_rate})

                # Generate summary with the current batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x_rgb: rgb_img_batch,
                                                            x_depth: depth_img_batch,
                                                            y: label_batch, keep_prob: 1.})

                    writer.add_summary(s, epoch*train_batches_per_epoch + step)

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(validation_init_op)
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            rgb_img_batch, depth_img_batch, label_batch = sess.run(next_batch)

            acc = sess.run(accuracy, feed_dict={x_rgb: rgb_img_batch,
                                                x_depth: depth_img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       test_acc))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
