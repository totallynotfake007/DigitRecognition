from PIL import Image
from PIL import ImageEnhance
import numpy as np
import PIL.ImageOps

size = 28, 28

brightness = 2.0

image_file = Image.open("./images/8.jpg")



enhancer = ImageEnhance.Brightness(image_file)
bright = enhancer.enhance(brightness)

bright.save("./images/8_bright.png")

new_image = Image.open("./images/8_bright.png")

grey_image = PIL.ImageOps.grayscale(new_image)

inv_image = PIL.ImageOps.invert(grey_image)
inv_image.save("./images/8_invert.png")
pix = list(inv_image.getdata())
print(pix)
inv_image.show()

