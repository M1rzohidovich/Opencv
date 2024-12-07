import cv2
import matplotlib.pyplot as plt
import numpy as np

# image = cv2.imread('C:/Users/hp/Videos/Captures/cat-2.jpg')
#
# plt.subplot(1, 2, 1)
# plt.title("Original")
# plt.imshow(image)
#
# # Adjust the brightness and contrast
# # Adjusts the brightness by adding 10 to each pixel value
# brightness = 10
# # Adjusts the contrast by scaling the pixel values by 2.3
# contrast = 2.3
# image2 = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)
#
# cv2.imwrite('modified_image.jpg', image2)
#
# plt.subplot(1, 2, 2)
# plt.title("Brightness & contrast")
# plt.imshow(image2)
# plt.show()

# Rasmni yuklash
# img = cv2.imread('C:/Users/hp/Videos/Captures/cat-4.jpg', 0)  # 0 ko‘rinishi grayscale formatga aylantirish uchun
#
# # Canny chegarani aniqlash
# edges = cv2.Canny(img, threshold1=50, threshold2=150)
#
# # Natijani ko'rsatish
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# plt.title("Asl rasm")
# plt.imshow(img, cmap='gray')
#
# plt.subplot(1, 2, 2)
# plt.title("Canny orqali chegara")
# plt.imshow(edges, cmap='gray')
# plt.show()

# # Import the Images module from pillow
# from PIL import Image
# import os
# import matplotlib.pyplot as plt
#
# # Rasm yo'lini kiriting
# image_path = "C:/Users/hp/Videos/Captures/cat-5.jpg"
#
# # Rasmni ochish
# image_file = Image.open(image_path)
#
# # Turli sifatlarda rasmni saqlash
# qualities = [95, 50, 25, 10, 1]
# images = []
# titles = []
#
# for quality in qualities:
#     output_path = f"C:/Users/hp/Videos/Captures/cat_quality_{quality}.jpg"
#     image_file.save(output_path, quality=quality)
#
#     # Saqlangan rasmni o'qiymiz
#     img = Image.open(output_path)
#     images.append(img)
#     size = os.path.getsize(output_path)
#     titles.append(f"Quality: {quality}\nSize: {size / 1024:.2f} KB")
#
# # Rasmni ko'rsatish
# fig, axes = plt.subplots(1, len(qualities), figsize=(15, 5))
# for ax, img, title in zip(axes, images, titles):
#     ax.imshow(img)
#     ax.set_title(title)
#     ax.axis('off')
#
# plt.tight_layout()
# plt.show()

#
# import cv2
# import matplotlib.pyplot as plt
#
# # Rasmni yuklash
# img = cv2.imread("C:/Users/hp/Videos/Captures/cat-5.jpg")
# print(type(img))
#
# # Rasm shakli (o'lchamlari)
# print("Shape of the image:", img.shape)
#
# # Rasmni kesish [rows, columns]
# crop = img[50:190, 100:270]
#
# # OpenCV BGR formatida rasm yuklaydi, Matplotlib esa RGB formatida ishlaydi.
# # Shuning uchun rasm ranglarini RGB formatiga o'zgartiramiz.
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#
# # Matplotlib yordamida rasmlarni konsolda ko'rsatish
# plt.figure(figsize=(10, 5))
#
# # Asosiy rasm
# plt.subplot(1, 2, 1)
# plt.imshow(img_rgb)
# plt.title("Original Image")
# plt.axis('off')
#
# # Kesilgan rasm
# plt.subplot(1, 2, 2)
# plt.imshow(crop_rgb)
# plt.title("Cropped Image")
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()


# import cv2
# import matplotlib.pyplot as plt
#
# # Rasmni yuklash
# img = cv2.imread("C:/Users/hp/Videos/Captures/cat-5.jpg")
# print(type(img))
#
# # Rasm shakli (o'lchamlari)
# print("Shape of the image:", img.shape)
#
# # Rasmni kesish [rows, columns]
# crop = img[50:190, 100:270]
#
# # Kesilgan rasmni kattalashtirish (upscale)
# # Interpolatsiya usuli: cv2.INTER_CUBIC (bu yuqori sifat beradi)
# upscaled_crop = cv2.resize(crop, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
#
# # OpenCV BGR formatida rasm yuklaydi, Matplotlib esa RGB formatida ishlaydi.
# # Shuning uchun rasm ranglarini RGB formatiga o'zgartiramiz.
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
# upscaled_crop_rgb = cv2.cvtColor(upscaled_crop, cv2.COLOR_BGR2RGB)
#
# # Matplotlib yordamida rasmlarni konsolda ko'rsatish
# plt.figure(figsize=(15, 5))
#
# # Asosiy rasm
# plt.subplot(1, 3, 1)
# plt.imshow(img_rgb)
# plt.title("Original Image")
# plt.axis('off')
#
# # Kesilgan rasm
# plt.subplot(1, 3, 2)
# plt.imshow(crop_rgb)
# plt.title("Cropped Image")
# plt.axis('off')
#
# # Kattalashtirilgan (upscaled) rasm
# plt.subplot(1, 3, 3)
# plt.imshow(upscaled_crop_rgb)
# plt.title("Upscaled Cropped Image")
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()


# import the necessary libraries
import numpy as np
import tensorflow as tf
from itertools import product

# set the param
plt.rc('figure', autolayout=True)
plt.rc('image', cmap='magma')

# define the kernel
kernel = tf.constant([[-1, -1, -1],
                    [-1,  8, -1],
                    [-1, -1, -1],
                   ])

# load the image
image = tf.io.read_file('Ganesh.jpg')
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[300, 300])

# plot the image
img = tf.squeeze(image).numpy()
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original Gray Scale image')
plt.show();


# Reformat
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)

# convolution layer
conv_fn = tf.nn.conv2d

image_filter = conv_fn(
    input=image,
    filters=kernel,
    strides=1, # or (1, 1)
    padding='SAME',
)

plt.figure(figsize=(15, 5))

# Plot the convolved image
plt.subplot(1, 3, 1)

plt.imshow(
    tf.squeeze(image_filter)
)
plt.axis('off')
plt.title('Convolution')

# activation layer
relu_fn = tf.nn.relu
# Image detection
image_detect = relu_fn(image_filter)

plt.subplot(1, 3, 2)
plt.imshow(
    # Reformat for plotting
    tf.squeeze(image_detect)
)

plt.axis('off')
plt.title('Activation')

# Pooling layer
pool = tf.nn.pool
image_condense = pool(input=image_detect,
                             window_shape=(2, 2),
                             pooling_type='MAX',
                             strides=(2, 2),
                             padding='SAME',
                            )

plt.subplot(1, 3, 3)
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.title('Pooling')
plt.show()

