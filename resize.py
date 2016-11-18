import os
from PIL import Image
from PIL import ImageEnhance
import numpy as np
import PIL.ImageOps


def resize(infile):
    size = 28, 28
    brightness = 2.0

    image_file = Image.open(infile)

    outfile = os.path.splitext(infile)[0] + ".thumbnail"
    try:
        image_file = image_file.resize(size, Image.ANTIALIAS)
        # im.thumbnail(size, Image.ANTIALIAS)
        enhancer = ImageEnhance.Brightness(image_file)
        bright = enhancer.enhance(brightness)
        image_file = PIL.ImageOps.invert(bright)
        #image_file = image_file.convert('LA')
        image_file = PIL.ImageOps.grayscale(image_file)
        pix = list(image_file.getdata())
        print(pix)
        image_file.save(outfile, "PNG")
        image_file.show()
        return outfile

    except IOError:
        print(IOError.__str__(IOError))
        print("cannot create thumbnail for '%s'" % infile)
        return "Error"

    #pix = list(inv_image.getdata())
    #print(pix)
    #inv_image.show()
