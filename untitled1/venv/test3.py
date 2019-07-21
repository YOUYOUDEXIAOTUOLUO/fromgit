import tensorflow as tf
from skimage import io
import matplotlib.pyplot as plt
import random

VOC_LABELS = {
    'none': (0, 'Background'),
    'plate': (1, 'plate')
}

bboxes = []

bboxes.append([0.3137931034482759, 0.1111111111111111, 0.36379310344827587, 0.33611111111111114])

def int64_feature(value):
    """
    生成整数型，浮点型和字符串型的属性
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

img = io.imread("D:/dataset/ccpd_select1/242&422_86&414_80&364_235&372-0_0_16_32_11_32_26.jpg")

figsize=(10,10)
linewidth = 1.5
fig = plt.figure(figsize=figsize)
plt.imshow(img)
height = img.shape[0]
width = img.shape[1]
colors = dict()
colors[0] = (random.random(), random.random(), random.random())
ymin = int(bboxes[0][0] * height)
xmin = int(bboxes[0][1] * width)
ymax = int(bboxes[0][2] * height)
xmax = int(bboxes[0][3] * width)
rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                     ymax - ymin, fill=False,
                     edgecolor=colors[0],
                     linewidth=linewidth)
plt.gca().add_patch(rect)
plt.show()
