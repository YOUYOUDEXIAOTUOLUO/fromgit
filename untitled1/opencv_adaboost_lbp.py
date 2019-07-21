import cv2
import os
import numpy as np
import con_reg as cr
from skimage.feature import multiblock_lbp
from skimage.feature import local_binary_pattern
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage import io, color
from skimage import draw, transform
from skimage.viewer import ImageViewer


# rectangle: x y w h
def extend_rec(image, rectangle):
    data = []
    labels = []
    shape = [1160, 720]
    ratio_pl = 44/17

    left_x = rectangle[0]
    top_y = rectangle[1]

    plate_w = rectangle[2]
    plate_h = rectangle[3]

    right_x = left_x + plate_w
    bottom_y = top_y + plate_h

    if plate_w/plate_h < ratio_pl:
        print(int(plate_w/plate_h))
        left_x = left_x - int((ratio_pl - plate_w/plate_h)*plate_h*0.6)
        right_x = right_x + int((ratio_pl - plate_w / plate_h) * plate_h*0.6)

    extend_x = 0.3 + 0.1 * np.random.uniform(-1, 1, 1)
    extend_y = 0.2 + 0.1 * np.random.uniform(-1, 1, 1)

    right_x = right_x + int(extend_x*plate_w)
    if right_x >= shape[1]:
        right_x = 719
    left_x = left_x - int((0.6 - extend_x)*plate_w)
    if left_x < 0:
        left_x = 0
    top_y = top_y - int(extend_y*plate_h)
    if top_y < 0:
        top_y = 0
    bottom_y = bottom_y + int((0.4 - extend_y) * plate_h)
    if bottom_y >= shape[0]:
        bottom_y = 719

    extend_img = image[top_y:bottom_y, left_x:right_x]
    extend_img_norm = transform.resize(extend_img, (145, 300))
    #io.imshow(extend_img)
    #io.show()
    # extend_w = right_x - left_x
    # extend_h = bottom_y - top_y

    vec = np.expand_dims(extend_img_norm, 2)
    data.append(vec)

    return np.array(data), [top_y, left_x, bottom_y - top_y, right_x - left_x]


plate_cascade = cv2.CascadeClassifier('D:/dataset/stage10-15000-20000/cascade.xml')
for root, director, files in os.walk("D:/dataset/ccpd_select2"):
    i = 0
    model = cr.init_mode()
    for file in files:
        if i < 500:
            img = cv2.imread("D:/dataset/ccpd_select2/"+file)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            plates, rw, rl = plate_cascade.detectMultiScale3(gray, 1.5, 5, cv2.CASCADE_SCALE_IMAGE,
                                                             outputRejectLevels=True)
            print(rl)
            print('\n')
            # for [lx, ly, lw, lh] in plates:
            #     cv2.rectangle(img, (lx, ly), (lx+lw, ly+lh), (0, 0, 255), 2)
            print(plates)
            # if num > 2:
            #     plates, num = plate_cascade.detectMultiScale(gray, 1.6, 5, cv2.CASCADE_SCALE_IMAGE)
            # if num >= 2:

            select = plates[0]

            rl = rl.tolist()
            maxl = rl.index(max(rl))
            print(maxl)
            k = 0
            tmp = plates[maxl][2]
            for plate in plates:
                if k <= maxl + 1 and k >= maxl - 1:
                    if plate[2] > tmp:
                        select = plate
                k+=1

            print(select)
            data, before_norm = extend_rec(gray, select)
            print('###')
            print(before_norm)

            ratio = cr.test_model(model, data)
            ratio = ratio[0]

            pl_ltx = int(before_norm[3]*ratio[0])
            pl_lty = int(before_norm[2]*ratio[1])

            pl_lbx = int(before_norm[3]*ratio[2])
            pl_lby = int(before_norm[2]*ratio[3])

            pl_rbx = int(before_norm[3]*ratio[4])
            pl_rby = int(before_norm[2]*ratio[5])

            pl_rtx = int(before_norm[3]*ratio[6])
            pl_rty = int(before_norm[2]*ratio[7])

            maxi = lambda x, y: x if x > y else y
            mini = lambda x, y: y if x > y else x
            print(pl_ltx, pl_lty, pl_lbx, pl_lby, pl_rbx, pl_rby, pl_rtx, pl_rty)
            x_t = mini(pl_ltx, pl_lbx) + before_norm[1]
            x_b = maxi(pl_rbx, pl_rtx) + before_norm[1]

            y_t = mini(pl_lty, pl_rty) + before_norm[0]
            y_b = maxi(pl_lby, pl_rby) + before_norm[0]
            print(x_b, x_t, y_b, y_t)

            cv2.rectangle(img, (x_t, y_t), (x_t + x_b - x_t, y_t + y_b - y_t), (0, 0, 255), 2)
            io.imshow(img)
            io.show()

        i += 1

