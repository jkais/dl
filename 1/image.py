import numpy as np
import scipy
import h5py
from PIL import Image
from scipy import ndimage

import matplotlib.pyplot as plt

f = h5py.File("mytestfile.hdf5", "w")

my_image = "cat.jpg"   # change this to the name of your image file

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
num_px = 64
my_image = scipy.misc.imresize(image, size=(num_px,num_px))#.reshape((1, num_px*num_px*3)).T

plt.imsave('thumb_cat.jpg', my_image)

print(my_image).shape
