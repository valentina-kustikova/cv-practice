
import numpy as np
import random

h, w = 1080, 1920


image = np.random.randint(0,256,(h,w,3),dtype=np.uint8)

xi=random.randint(0,w)
yi=random.randint(0,h)


print(image[4])



