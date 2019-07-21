# -*- coding：utf-8 -*-
# -*- author：zzZ_CMing  CSDN address:https://blog.csdn.net/zzZ_CMing
# -*- 2018/07/17; 13:18
# -*- python3.5
"""
特别注意: 17行VOC_LABELS标签要修改，189行的path地址要正确
"""

import os
import sys
import random
import numpy as np
from skimage import io
import tensorflow as tf
import xml.etree.ElementTree as ET

# 我的标签定义只有手表这一类，所以下面的VOC_LABELS要根据自己的图片标签而定，第一组'none': (0, 'Background')是不能删除的；
VOC_LABELS = {
    'none': (0, 'Background'),
    'plate': (1, 'plate')
}

# 图片和标签存放的文件夹.
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'

# 随机种子.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 3  # 每个.tfrecords文件包含几个.xml样本



def int64_feature(value):
    """
    生成整数型，浮点型和字符串型的属性
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def single_image_content(img_file, target_xy, isNone):

    # Read the image file.
    img_data = tf.gfile.FastGFile(img_file, 'rb').read()

    #Load the shape
    print(img_file)
    img = io.imread(img_file)
    shape = list(img.shape)


    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []

    if not isNone:

        labels.append(int(VOC_LABELS['plate'][0]))
        labels_text.append('plate'.encode('ascii'))  # 变为ascii格式

    else:
        labels.append(int(VOC_LABELS['none'][0]))
        labels_text.append('none'.encode('ascii'))  # 变为ascii格式

    difficult.append(0)
    truncated.append(0)

    a = float(target_xy[0]) / shape[0]  #y_min
    b = float(target_xy[1]) / shape[1]
    a1 = float(target_xy[2]) / shape[0]
    b1 = float(target_xy[3]) / shape[1]
    if a > 1 or b > 1 or a1 > 1 or b1 >1:
        print("ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR!")
        print(target_xy)
        print(img_file)
    a_e = a1 - a
    b_e = b1 - b
    if abs(a_e) < 1 and abs(b_e) < 1:
        bboxes.append((a, b, a1, b1))
    return img_data, shape, bboxes, labels, labels_text, difficult, truncated


def convert_to_tfrecord(image_data, isNone, labels, labels_text, bboxes, shape, difficult, truncated):
    """
    转化样例
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []

    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)}))
    return example


def traverse(dataset_dir, tfrecord_dir, isNone, name='voc_train', shuffling=False):

    for root, director, files in os.walk(dataset_dir):
        i = 0

        for file in files:

            substr = files[i].split("-")
            subsubstr = substr[0].split("_")

            target_xy = []

            if not isNone:

                left_top = subsubstr[2]
                left_bottom = subsubstr[1]
                right_bottom = subsubstr[0]
                right_top = subsubstr[3]

                left_top_x = int(left_top.split("&")[0])
                left_top_y = int(left_top.split("&")[1])

                left_bottom_x = int(left_bottom.split("&")[0])
                left_bottom_y = int(left_bottom.split("&")[1])

                right_bottom_x = int(right_bottom.split("&")[0])
                right_bottom_y = int(right_bottom.split("&")[1])

                right_top_x = int(right_top.split("&")[0])
                right_top_y = int(right_top.split("&")[1])

                maxi = lambda x, y: x if x > y else y
                mini = lambda x, y: y if x > y else x

                target_xy.append(mini(left_top_y, right_top_y))
                target_xy.append(mini(left_top_x, left_bottom_x))
                target_xy.append(maxi(left_bottom_y, right_bottom_y))
                target_xy.append(maxi(right_bottom_x, right_top_x))

            else:
                target_xy = [0, 0, 0, 0]

            print(target_xy)

            image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
                single_image_content(os.path.join(dataset_dir, file), target_xy, isNone)

            example = convert_to_tfrecord(image_data, isNone, labels, labels_text,
                              bboxes, shape, difficult, truncated)

            file_name = 'voc_2007_none_%03d.tfrecord' % i
            tf_filename = os.path.join(tfrecord_dir, file_name)

            tf.python_io.TFRecordWriter(tf_filename).write(example.SerializeToString())

            i += 1

    print('\nFinished converting the Pascal VOC dataset!')


def create_tfrecord_set():

    dataset_dir = "D:/dataset/ccpd_np/"
    tfrecord_dir = "D:/dataset/tfrecords0/"
    traverse(dataset_dir, tfrecord_dir, 1)


create_tfrecord_set()
