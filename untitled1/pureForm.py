
from skimage.feature import multiblock_lbp
from skimage.feature import local_binary_pattern
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage import io, color
from skimage import draw, transform
from skimage.viewer import ImageViewer
from sklearn import ensemble
from sklearn import tree
from sklearn.externals import joblib
from sklearn.datasets import make_gaussian_quantiles
from sklearn import feature_extraction as fex
import sklearn
import skimage
import numpy as np
import matplotlib.pyplot as plt
import os


def file_parase(file_dir):
    desc = open("D:/dataset/sample_key.dat", 'w')
    i = 0
    for root, director, files in os.walk(file_dir):
        print(files)
        for file in files:

            image = io.imread(file_dir + file)
            #img_gray = color.rgb2gray(image)
            img_union = transform.resize(image, [80, 190])
            io.imsave(file_dir + file, img_union)
            #substr = files[i].split("-")
            #content = substr[3] + '-' + substr[4]
            #os.rename(os.path.join(file_dir, file), os.path.join(file_dir, content + ".jpg"))
            content = file + ' 1 0 0 190 80' + '\n'
            print(content)
            desc.write(content)
            desc.flush()
            i += 1


feature_types = ['type-2-x', 'type-2-y',
                 'type-3-x', 'type-3-y',
                 'type-4']


def lbp_feature_fit(pos_dir, neg_dir):
    para = [[]]
    y = []
    classi = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2), algorithm="SAMME.R", n_estimators=50,
                                         learning_rate=1.2)
    for root, director, files in os.walk(pos_dir):
        print(files)
        i = 0
        for file in files:
            i += 1
            if (i > 50):
                break
            print(file)
            image = io.imread(pos_dir + file)
            img_gray = color.rgb2gray(image)
            img_lbp = local_binary_pattern(img_gray, 16, 2)
            print(img_lbp)
            para = np.array([img_lbp])
            # y = np.array(1)
            classi.fit(para, [0])

    for root, director, files in os.walk(neg_dir):
        print(files)
        i = 0
        for file in files:
            i += 1
            if (i > 50):
                break
            print(file)
            image = io.imread(neg_dir + file)

            img_gray = color.rgb2gray(image)
            img_lbp = local_binary_pattern(img_gray, 16, 2)

            para = np.array([img_lbp])
            # y = np.array(0)
            classi.fit(para, [0])

    joblib.dump(classi, "D:/dataset/lbpmodel.m")


def pl_cut_out(file_dir, des_dir):

    desc = open("D:/dataset/key2.dat", 'w')
    i = 6230
    for root, director, files in os.walk(file_dir):

        for file in files:
            if i < 15000:
                print(file)
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

                image = io.imread(file_dir+file)
                img_gray = color.rgb2gray(image)

                print(x_t, x_b, y_t, y_b)
                lp = img_gray[y_t:y_b, x_t:x_b]

                des = des_dir+"plate"+str(i)+".jpg"

                io.imsave(des_dir+"plate"+str(i)+".jpg", lp)

            i += 1


def haar_feature_fit(pos_dir, neg_dir):
    para = [[]]
    y = []
    classi = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2), algorithm="SAMME.R", n_estimators=50,
                                         learning_rate=1.2)
    for root, director, files in os.walk(pos_dir):
        print(files)
        i = 0
        for file in files:
            i += 1
            if(i > 50):
                break
            print(file)
            image = io.imread(pos_dir+file)
            img_gray = color.rgb2gray(image)
            img_union = transform.resize(img_gray, (190, 100))
            coord, ft = haar_like_feature_coord(100, 60, feature_types[0])
            feature = haar_like_feature(img_union, 0, 0, 100, 60, ft, coord)

            para = np.array([feature])
            #y = np.array(1)
            classi.fit(para, [0])

    for root, director, files in os.walk(neg_dir):
        print(files)
        i = 0
        for file in files:
            i += 1
            if (i > 50):
                break
            print(file)
            image = io.imread(neg_dir + file)

            img_gray = color.rgb2gray(image)
            coord, ft = haar_like_feature_coord(100, 60, feature_types[0])
            feature = haar_like_feature(img_gray, 0, 0, 100, 60, ft, coord)

            para = np.array([feature])
            #y = np.array(0)
            classi.fit(para, [0])

    joblib.dump(classi, "D:/dataset/haarmodel.m")
    image = io.imread("D:/dataset/ccpd_select1/242&422_86&414_80&364_235&372-0_0_16_32_11_32_26.jpg")
    img_gray = color.rgb2gray(image)
    coord, ft = haar_like_feature_coord(100, 60, feature_types[0])
    feature = haar_like_feature(img_gray, 0, 0, 100, 60, ft, coord)

    test = np.array([feature])
    res = classi.predict(test)

    print(res)


#pl_cut_out("D:/dataset/ccpd_select1/", "D:/dataset/pos/")

def detection(img):
    width = img.shape[0]
    height = img.shape[1]
    print(height)
    ptr_w = 0
    ptr_h = 0
    classi = joblib.load("D:/dataset/haarmodel.m")

    while ptr_h < height:
        while ptr_w < width:
            part_img = img[ptr_w:ptr_w+60, ptr_h:ptr_h+100]

            img_gray = color.rgb2gray(part_img)
            io.imshow(img_gray)
            io.show()
            coord, ft = haar_like_feature_coord(100, 60, feature_types[0])
            feature = haar_like_feature(img_gray, 0, 0, 100, 60, ft, coord)
            test = np.array([feature])
            res = classi.predict(test)
            if res == 1:
                print("PL detected")
            else:
                print("NOT detected"+str(ptr_w)+"_"+str(ptr_h))
            if (ptr_w+90) == width:
                ptr_w = width+1
            else:
                ptr_w += 70
                if (ptr_w+70) > width:
                    ptr_w = width - 90
        ptr_w = 0
        if (ptr_h+100) == height:
            ptr_h = height+1
        else:
            ptr_h += 100
            if (ptr_h + 100) > height:
                print("end")
                ptr_h = height - 100


def data_augmentation(folder, des_folder):
    desc = open("D:/dataset/sample_neg.dat", 'w')
    i = 0
    for root, director, files in os.walk(folder):
        sub = []
        for file in files:
            image = io.imread(folder + file)
            print(file)
            sub.clear()
            sub.append(image[0:799, 0:499])
            sub.append(image[200:999, 100:599])
            sub.append(image[0:799, 220:719])
            sub.append(image[200:999, 0:499])
            sub.append(image[360:1159, 0:499])
            sub.append(image[160:959, 0:499])
            sub.append(image[360:1159, 220:719])
            sub.append(image[160:959, 220:719])
            img_gray = color.rgb2gray(image)
            #img_union = transform.resize(image, [80, 190])
            k = 0
            for k in range(8):
                print(k)
                file_new = des_folder + str(i + k) + '.jpg'
                print(file_new)
                io.imsave(file_new, sub[k])
                content = str(i + k) + '.jpg' + '\n'
                desc.write(content)
                desc.flush()
            i += 8


#lbp_feature_fit("D:/dataset/pos/", "D:/dataset/ccpd_dataset/ccpd_np/")
#joblib.load("D:/dataset/haarmodel.m")
#image = io.imread("D:/dataset/ccpd_select1/242&422_86&414_80&364_235&372-0_0_16_32_11_32_26.jpg")
#detection(image)

#pl_cut_out('D:/dataset/ccpd_select2/', 'D:/dataset/pos/')
#file_parase("D:/dataset/pos/")
data_augmentation('D:/dataset/ccpd_np/', 'D:/dataset/neg/')
# viewer = ImageViewer(image2)
# viewer.show()
