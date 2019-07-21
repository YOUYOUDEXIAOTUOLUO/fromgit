from skimage import data
from matplotlib import pyplot as plt
from skimage.transform import integral_image
from skimage.feature import multiblock_lbp
from skimage.feature import draw_multiblock_lbp

test_img = data.coins()

plt.imshow(test_img, interpolation='nearest')

plt.show()

int_img = integral_image(test_img)

lbp_code = multiblock_lbp(int_img, 0, 0, 190, 90)

img = draw_multiblock_lbp(test_img, 0, 0, 190, 90,
                          lbp_code=lbp_code, alpha=0.5)


plt.imshow(img, interpolation='nearest')

plt.show()