import sys
sys.path.append('/home/jinyong/projects/datasets/')
from utils_packages_me import me
import pandas as pd
import numpy as np

# f = pd.read_csv('/shares/crowd_counting/C3/ProcessedData/UCF_CC_50/den/1/3.csv', sep=',', header=None)
f = pd.read_csv('/shares/crowd_counting/C3/ProcessedData/shanghaitech_part_B/test/den/1.csv').values

# print(f)
# print(f.values)

f = f.astype(np.float32, copy=False)
# print(f)
me.np2jpg(f, 'b.jpg')
# img = Image.fromarray(f)

# img.save('a.jpg')
