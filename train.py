import os
from Model.xception import xception, train_model
from LoadData.LoadData import InputTrainImg, InputTestImg

btch = 32
epoch = 1000
width = 128

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

datasets = InputTrainImg('D:/data/FNHHCC/model', width)
train_x, train_y = datasets.load_train_data()

dat = InputTestImg('D:/data/FNHHCC/test', width)
test_x, test_y = dat.load_test_data()
train_model(train_x, train_y, test_x, test_y, epoch, btch, width)
