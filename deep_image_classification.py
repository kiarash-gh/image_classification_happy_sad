import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt



cpus = tf.config.experimental.list_physical_devices('CPU')

# for cpu in cpus:
#     tf.config.experimental.set_memory_growth(cpu, True)


data_dir = 'data'
image_exts = [ 'jpeg', 'jpg', 'bmp', 'png']


for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print(f'image not in ext list {image_path}')
                os.remove(image_path)

        except Exception as ex:
            print(f'issue with image{image_path}')



# load data
data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

fig, ax  = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# preprocessing 

data = data.map(lambda x,y: (x/255, y))

data.as_numpy_iterator().next()

print(len(data))