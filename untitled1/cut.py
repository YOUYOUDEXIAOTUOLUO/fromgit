from skimage import color, io

img = io.imread("D:/dataset/pos/plate3.jpg")
img_gray = color.rgb2gray(img)

# 二值化
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i, j] <= 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1

col_sum = 0

left = [0, 0, 0, 0, 0, 0, 0]

k = 0
for j in range(cols):
    for i in range(rows):
        col_sum += img_gray[i][j]

    if col_sum < 15 and (j - left[k]) > 20 and k < 6:
        left[k + 1] = j
        k += 1
    col_sum = 0

print(left)
io.imshow(img_gray)
io.show()
