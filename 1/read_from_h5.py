import h5py

with h5py.File('images.hdf5', 'r') as f:
        test_data = f['parsed_images'][()]
        image_classes = f['image_classes'][()]
print(test_data)
print(image_classes[0])
