from Model.xception import xception
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, Sequential, optimizers
from LoadData.LoadData import InputTrainImg, InputTestImg
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet121
from LoadData.LoadData import InputTrainImg, InputTestImg
from Model.xception import xception
from tensorflow.keras import layers, Sequential, optimizers


width = 128
shape = width, width, 3
model = xception(shape, include_top=True)
optimizer = optimizers.Adam(learning_rate=1e-5)  # 学习率

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

base_model2 = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(width,width,3))
base_model2.trainable = False
x = base_model2.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64,activation='relu')(x)
x = tf.keras.layers.Dense(1,activation="sigmoid")(x)
model2 = tf.keras.Model(inputs=base_model2.input, outputs=x)
model2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

base_model4 = DenseNet121(weights='imagenet', include_top=False, input_shape=(width, width, 3))
base_model4.trainable = False
x = base_model4.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64,activation='relu')(x)
x = tf.keras.layers.Dense(1,activation="sigmoid")(x)
model4 = tf.keras.Model(inputs=base_model4.input, outputs=x)


model4.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

base_model7 = InceptionV3(weights='imagenet', include_top=False, input_shape=(width, width, 3))
base_model7.trainable = False
x = base_model7.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64,activation='relu')(x)
x = tf.keras.layers.Dense(1,activation="sigmoid")(x)
model7 = tf.keras.Model(inputs=base_model7.input, outputs=x)
model7.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

base_model8 = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(width, width, 3))
base_model8.trainable = False
x = base_model8.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64,activation='relu')(x)
x = tf.keras.layers.Dense(1,activation="sigmoid")(x)
model8 = tf.keras.Model(inputs=base_model8.input, outputs=x)


model8.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.load_weights('Liverre_40_0.88')
model2.load_weights('Liver2_3_1.00')
model4.load_weights('Liver4_3_1.00')
model7.load_weights('Liver7_4_1.00')
model8.load_weights('Liver8_36_0.89')

# dat = InputTestImg('D:/data/liverUSzhang/test/', width)
dat = InputTestImg('D:/data/LiverUSZhang/model', width)
test_x, test_y = dat.load_test_data()

score = model.evaluate(test_x, test_y)
score2 = model2.evaluate(test_x, test_y)
score4 = model4.evaluate(test_x, test_y)
score7 = model7.evaluate(test_x, test_y)
score8 = model8.evaluate(test_x, test_y)
pre_y = model.predict(test_x)
pre_y[pre_y > 0.5] = 1
pre_y[pre_y < 0.5] = 0
pre_y2 = model2.predict(test_x)
pre_y2[pre_y2 > 0.5] = 1
pre_y2[pre_y2 < 0.5] = 0
pre_y4 = model4.predict(test_x)
pre_y4[pre_y4 > 0.5] = 1
pre_y4[pre_y4 < 0.5] = 0
pre_y7 = model7.predict(test_x)
pre_y7[pre_y7 > 0.5] = 1
pre_y7[pre_y7 < 0.5] = 0
pre_y8 = model8.predict(test_x)
pre_y8[pre_y8 > 0.5] = 1
pre_y8[pre_y8 < 0.5] = 0

c = confusion_matrix(test_y, pre_y)
c2 = confusion_matrix(test_y, pre_y2)
c4 = confusion_matrix(test_y, pre_y4)
c7 = confusion_matrix(test_y, pre_y7)
c8 = confusion_matrix(test_y, pre_y8)
print(c)
print(c2)
print(c4)
print(c7)
print(c8)

plt.figure()
classes = ['FNH', 'HCC']

plt.imshow(c, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
plt.title('Confusion Matrix of Liver Tumour')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=-45)
plt.yticks(tick_marks, classes)

thresh = c.max() / 2.
iters = np.reshape([[[i, j] for j in range(2)] for i in range(2)], (c.size, 2))
for i, j in iters:
    plt.text(j, i, format(c[i, j]))  # 显示对应的数字

plt.ylabel('Real label')
plt.xlabel('Prediction')
plt.tight_layout()
plt.savefig('CHCHCCICC.svg', format='svg')
plt.show()

print(precision_score(test_y, pre_y))
print(recall_score(test_y, pre_y))
print(f1_score(test_y, pre_y))
print(precision_score(test_y, pre_y2))
print(recall_score(test_y, pre_y2))
print(f1_score(test_y, pre_y2))
print(precision_score(test_y, pre_y4))
print(recall_score(test_y, pre_y4))
print(f1_score(test_y, pre_y4))
print(precision_score(test_y, pre_y7))
print(recall_score(test_y, pre_y7))
print(f1_score(test_y, pre_y7))
print(precision_score(test_y, pre_y8))
print(recall_score(test_y, pre_y8))
print(f1_score(test_y, pre_y8))
# print(test_y)
# print(pre_y)
print(roc_auc_score(test_y, model.predict(test_x)))
print(roc_auc_score(test_y, model2.predict(test_x)))
print(roc_auc_score(test_y, model4.predict(test_x)))
print(roc_auc_score(test_y, model7.predict(test_x)))
print(roc_auc_score(test_y, model8.predict(test_x)))
# print(train_x.shape)
# print(final_pre)
# print(test_y)
# print(test_z[test_y == 0])
# print("test_score: ", score)
# print(test_z[final_pre != test_y])
# print(test_y[final_pre != test_y])
# print(test_z[final_pre != test_y].shape)
# First aggregate all false positive rates

y = test_y
pred_y = model.predict(test_x)
pred_y2 = model2.predict(test_x)
pred_y4 = model4.predict(test_x)
pred_y7 = model7.predict(test_x)
pred_y8 = model8.predict(test_x)
# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)

# Compute ROC curve and ROC area for each class

fpr, tpr, _ = roc_curve(y, pred_y)
fpr2, tpr2, _ = roc_curve(y, pred_y2)
fpr4, tpr4, _ = roc_curve(y, pred_y4)
fpr7, tpr7, _ = roc_curve(y, pred_y7)
fpr8, tpr8, _ = roc_curve(y, pred_y8)

roc_auc = auc(fpr, tpr)
roc_auc2 = auc(fpr2, tpr2)
roc_auc4 = auc(fpr4, tpr4)
roc_auc7 = auc(fpr7, tpr7)
roc_auc8 = auc(fpr8, tpr8)


plt.figure()


plt.plot(
    fpr,
    tpr,
    label="ROC curve of Our Method(area = {0:0.2f})".format(roc_auc),
    color="navy",
    linestyle="-",
    linewidth=2,
)
plt.plot(
    fpr2,
    tpr2,
    label="ROC curve of MobileNet(area = {0:0.2f})".format(roc_auc2),
    color="aqua",
    linestyle="-",
    linewidth=2,
)
plt.plot(
    fpr8,
    tpr8,
    label="ROC curve of ResNet50(area = {0:0.2f})".format(roc_auc8),
    color="lightgreen",
    linestyle="-",
    linewidth=2,
)
plt.plot(
    fpr4,
    tpr4,
    label="ROC curve of DenseNet121(area = {0:0.2f})".format(roc_auc4),
    color="darkorange",
    linestyle="-",
    linewidth=2,
)

plt.plot(
    fpr7,
    tpr7,
    label="ROC curve of InceptionV3(area = {0:0.2f})".format(roc_auc7),
    color="deeppink",
    linestyle="-",
    linewidth=2,
)


lw = 2
# colors = cycle(["aqua", "darkorange", "cornflowerblue"])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(
#         fpr[i],
#         tpr[i],
#         color=color,
#         lw=lw,
#         label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
#     )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC of Liver Cancer in test Cohort")
plt.legend(loc="lower right")
plt.savefig('ROCtrain.svg',format='svg')
plt.show()

pd.DataFrame(y).to_csv('y.csv')
pd.DataFrame(pred_y).to_csv('1.csv')
pd.DataFrame(pred_y2).to_csv('2.csv')
pd.DataFrame(pred_y4).to_csv('4.csv')
pd.DataFrame(pred_y7).to_csv('7.csv')
pd.DataFrame(pred_y8).to_csv('8.csv')


