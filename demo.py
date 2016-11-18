from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.models import model_from_json
from PIL import Image

import scipy.misc as smp
import numpy as np
import resize

image_path = input("Enter image name: ")
#a = image_path.split('/')
#image = a[len(a)-1]
#image_path = "/images/"+image_path


model = model_from_json(open('my_model_architecture.json').read())
model.load_weights('my_model_weights.h5')

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


ima = Image.open(image_path)
pixels = list(ima.getdata())


width, height = ima.size
pixels = np.asarray(pixels)
pixels = pixels.reshape(1, 784)


ndigits = 1
images = pixels.reshape([ndigits, 28, 28])
imagesums = np.sum(np.sum(images, axis=1), axis=1)


indices = np.arange(28)
X, Y = np.meshgrid(indices, indices)

centroidx = np.sum(images * X) / imagesums

centroidy = np.sum(images * Y) / imagesums

pixels = pixels/255


print(pixels)

ans = model.predict(x=pixels)
print("\n\n", ans)
