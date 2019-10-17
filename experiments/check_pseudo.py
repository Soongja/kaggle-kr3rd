import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# frame = pd.read_csv('se_resnext_50_256_480_5fold_2.csv')
frame = pd.read_csv('se_resnext_50_256_480_5fold_all.csv')
print('len:', len(frame), '*0.95=', len(frame)*0.95)

pseudo = frame[frame['confidence']>0.50]
pseudo0 = frame[frame['confidence']>0.60] # 4524
pseudo1 = frame[frame['confidence']>0.70] # 4062
pseudo2 = frame[frame['confidence']>0.80]
print(len(pseudo))
print(len(pseudo0))
print(len(pseudo1))
print(len(pseudo2))

# plt.hist(frame['confidence'], bins=50)
# plt.show()