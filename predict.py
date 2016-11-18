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
import preprocess

image_path = input("Enter image name: ") #Take
#a = image_path.split('/')
#image = a[len(a)-1]
#image_path = "/images/"+image_path
#converted = resize.resize(image_path)
pixels = preprocess.preprocess(image_path)

model = model_from_json(open('my_model_architecture.json').read())
model.load_weights('my_model_weights.h5')

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


#ima = Image.open(converted)
#pixels = np.array(ima.getdata())
#print(pixels)

#pixels = np.array(pixels)


#pixels = pixels[:len(pixels), :1]
#print(pixels.shape)
pixels = pixels.reshape(1, 784)
#pixels = pixels.astype('float32')
#pixels /= 255


ans = model.predict(x=pixels)
main = np.asarray(ans)
print("\n\n", ans)
print("\n\nThe digit is: ", main.argmax())
