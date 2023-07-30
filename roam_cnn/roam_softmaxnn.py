import numpy as np
from torch import nn

from utils.decorators import init_net


print('collecting data...')
small_data = False
feature_dir = '../vortex/small_feature' if small_data else '../vortex/0823SPECKLE'
walkGen = os.walk(feature_dir)
feature_data = []
for _, __, file_names in walkGen:
    file_names = sorted(file_names, key=lambda name: int(name.split(".")[0]))  # 给文件名排序！
    featureImg_array = map(np.array, [plt.imread(feature_dir + '/' + file_name) for file_name in file_names])
    featureImgs = [featureImg for featureImg in featureImg_array]
    feature_data += featureImgs
    del featureImgs, file_names, featureImg_array

label_fileName = '../vortex/small_labels.csv' if small_data else '../vortex/labels.csv'
label_colNames = ['OAM1', 'OAM2']
label_data = pd.read_csv(label_fileName, names=label_colNames).values