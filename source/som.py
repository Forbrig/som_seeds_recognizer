import matplotlib.pyplot as pl
import numpy as np

# to print all np.array
#np.set_printoptions(threshold=np.nan)

file = open("seeds_dataset.txt")
data = file.readlines()

data = [x.replace('\n', '').replace('\t', ' ').replace('  ', ' ') for x in data]
features = np.array([line.split(' ')[0:8] for line in data], dtype = float)

#print(features)
