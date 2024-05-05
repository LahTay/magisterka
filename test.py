import numpy as np


list1 = [np.full((10,), 1), np.full((10,), 2)]

test = np.stack(list1, axis=1)
a=0