from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from skimage import io, color, transform
import numpy as np
import os


def gen_rectangle(file):
    substr = file.split("-")
    subsubstr = substr[0].split("_")

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

    x_t = mini(left_top_x, left_bottom_x)
    x_b = maxi(right_bottom_x, right_top_x)

    y_t = mini(left_top_y, right_top_y)
    y_b = maxi(left_bottom_y, right_bottom_y)
    return [x_t, x_b, y_t, y_b], [[left_top_x, left_top_y], [left_bottom_x, left_bottom_y],
                                  [right_bottom_x, right_bottom_y], [right_top_x, right_top_y]]


def gen_train_set(folder, num):
    data = []
    labels = []
    i = 0
    shape = [1160, 720]
    for root, director, files in os.walk(folder):
        for file in files:
            if i < num:
                print(i)
                image = io.imread(folder + file)
                rectangle, plate = gen_rectangle(file)
                img_gray = color.rgb2gray(image)
                #img_gray = img_gray.astype(np.float32)/255
                plate_w = rectangle[1] - rectangle[0]
                plate_h = rectangle[3] - rectangle[2]

                extend_x = 0.3 + 0.1 * np.random.uniform(-1, 1, 1)
                extend_y = 0.3 + 0.1 * np.random.uniform(-1, 1, 1)

                right_x = rectangle[1] + int(extend_x*plate_w)
                if right_x >= shape[1]:
                    right_x = 719
                left_x = rectangle[0] - int((0.6 - extend_x)*plate_w)
                if left_x < 0:
                    left_x = 0
                top_y = rectangle[2] - int(extend_y*plate_h)
                if top_y < 0:
                    top_y = 0
                bottom_y = rectangle[3] + int((0.6 - extend_y) * plate_h)
                if bottom_y >= shape[0]:
                    bottom_y = 1159
                # print([top_y, bottom_y, left_x, right_x])

                extend_img = img_gray[top_y:bottom_y, left_x:right_x]
                extend_img = transform.resize(extend_img, (145, 300))
                #io.imshow(extend_img)
                #io.show()
                extend_w = right_x - left_x
                extend_h = bottom_y - top_y

                plate_norm_ltx = (plate[0][0] - left_x)/extend_w
                plate_norm_lty = (plate[0][1] - top_y) / extend_h

                plate_norm_lbx = (plate[1][0] - left_x) / extend_w
                plate_norm_lby = (plate[1][1] - top_y) / extend_h

                plate_norm_rbx = (plate[2][0] - left_x) / extend_w
                plate_norm_rby = (plate[2][1] - top_y) / extend_h

                plate_norm_rtx = (plate[3][0] - left_x) / extend_w
                plate_norm_rty = (plate[3][1] - top_y) / extend_h
                vec = np.expand_dims(extend_img, 2)
                data.append(vec)
                labels.append([plate_norm_ltx, plate_norm_lty, plate_norm_lbx, plate_norm_lby,
                               plate_norm_rbx, plate_norm_rby, plate_norm_rtx, plate_norm_rty])

                i += 1

    return np.array(data), np.array(labels)


def get_model(path):
    model = Sequential()
    model.add(Convolution2D(1, (7, 7), padding='valid', input_shape=(145, 300, 1), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Convolution2D(1, (3, 3), padding='valid', strides=(2, 2)))
    model.add(Convolution2D(32, (5, 5), padding='valid', strides=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    # model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='normal'))
    model.add(Activation('relu'))
    model.add(Dense(8, kernel_initializer='normal'))
    model.add(Activation('relu'))

    if path != 'null':
        model.load_weights(path)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model


def train_model():
    model = get_model('D:/dataset/regression.m')
    t_data, t_labels = gen_train_set('D:/dataset/ccpd_select2/', 10000)
    model.fit(t_data, t_labels, batch_size=50, nb_epoch=50)
    model.save('D:/dataset/regression.m')


def test_model(model, data):

    points = model.predict(data)
    print(points)
    return points


def init_mode():
    model = get_model('D:/dataset/regression.m')
    return model

# train_model()
# t_data, t_labels = gen_train_set('D:/dataset/ccpd_select1/', 1)
# test_model(t_data)
