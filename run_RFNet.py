import os
from time import time
import time as T



print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('RFNet train')
tic = time()
os.system('python train_RFNet.py')
hours, rem = divmod(time()-tic, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('RAFNet test')
tic = time()
os.system('python test_RFNet.py')
hours, rem = divmod(time()-tic, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

