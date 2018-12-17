from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import png

def read_and_decode(filename_queue):
 reader = tf.TFRecordReader()
 _, serialized_example = reader.read(filename_queue)
 features = tf.parse_single_example(
  serialized_example,
  # Defaults are not specified since both keys are required.
  features={
      'label': tf.FixedLenFeature([], tf.int64),
      'label_normal': tf.FixedLenFeature([], tf.int64),
      'image': tf.FixedLenFeature([], tf.string)
  })
 # extract the data
 label = features["label_normal"]
 image = tf.decode_raw(features['image'], tf.uint8)

 # reshape and scale the image
 image = tf.reshape(image, [299, 299])
 return image, label


def get_all_records(files):
 with tf.Session() as sess:
   filename_queue = tf.train.string_input_producer(files)
   image, label = read_and_decode(filename_queue)
   init_op = tf.initialize_all_variables()
   sess.run(init_op)
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(coord=coord)
   i = 1
   sum_imgs = 0
   while sum_imgs < 55890:
     example, l = sess.run([image, label])
     if l == 1:
       filename = "cancer_imgs/cancer_" + str(i) + ".png"
       #plt.imshow(example)
       #plt.savefig(filename, bbox_inches='tight', pad_inches=0)
       png.from_array(example, "L").save(filename)
       i += 1
     sum_imgs += 1
     

get_all_records(["training10_0.tfrecords", "training10_1.tfrecords", "training10_2.tfrecords", "training10_3.tfrecords", "training10_4.tfrecords"])