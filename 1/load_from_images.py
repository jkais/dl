import numpy as np
import scipy
from scipy import ndimage
import h5py
import glob
import matplotlib.pyplot as plt
import re

pixels = 64
data_file = "images.hdf5"
f = h5py.File(data_file, "w")
images = glob.glob('images/*.jpg')
images_count = len(images)
parsed_images = np.empty((images_count, pixels, pixels, 3))
image_classes = np.empty((images_count))

for index, filename in enumerate(images):
    print("Parsing " + str(index) + ": " + filename + "...")
    image = np.array(ndimage.imread(filename, flatten=False))
    resized_image = scipy.misc.imresize(image, size=(pixels, pixels))
    parsed_images[index] = resized_image
    print("added pixels to test array")
    if re.search(r'cat', filename) is None:
        print("According to the filename it's not a cat.")
        image_classes[index] = False
    else:
        print("According to the filename it's a cat.")
        image_classes[index] = True
    resized_image_filename = re.sub("^images/", "images/thumbs/", filename)
    plt.imsave(resized_image_filename, resized_image)
    print("generated thumbnail " + resized_image_filename)

print("image data is of shape " + str(parsed_images.shape))
f.create_dataset("parsed_images", data=parsed_images)
f.create_dataset("image_classes", data=image_classes)
f.close()
print("wrote data to " + data_file)
