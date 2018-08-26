from numpy import linalg as LA
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

file = '/mnt/fcav/self_training/object_detection/lowerbound/prediction/test/cityscapes_caronly_val/generalized_rcnn/ious_checkpoint16999.xlsx'
output_dir = '/mnt/fcav/self_training/object_detection/lowerbound/prediction/test/cityscapes_caronly_val/localization_error'
dataset = 'GTA'
score_thr = 0.9
iou_thr = 0.7
# area_thr = 10000

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

xl = pd.ExcelFile(file)
df = xl.parse('Sheet1')
ious = df.iou._values
score = df.score._values
area = df.area._values
bbox_x = df.bbox_x._values
bbox_y = df.bbox_y._values
bbox_w = df.bbox_w._values
bbox_h = df.bbox_h._values
gt_bbox_x = df.gt_bbox_x._values
gt_bbox_y = df.gt_bbox_y._values
gt_bbox_w = df.gt_bbox_w._values
gt_bbox_h = df.gt_bbox_h._values

loc_error = []
if len(bbox_x) != len(gt_bbox_x):
    raise ValueError('detection number is not compatible to ground truth number')

for i in range(len(bbox_x)):
    err_temp = []
    error = [bbox_x[i] - gt_bbox_x[i],
             bbox_y[i] - gt_bbox_y[i],
             bbox_w[i] - gt_bbox_w[i],
             bbox_h[i] - gt_bbox_h[i]]
    for err in error:
        if abs(err) < 1:
            err_temp.append(0.5 * err ** 2)
        else:
            err_temp.append(abs(err) - 0.5)
    loc_error.append(sum(err_temp))


score_ind = np.where(score > score_thr)
iou_ind = np.where(ious > iou_thr)
# inds = score_ind[0]
inds = [val for val in score_ind[0] if val in iou_ind[0]]
loc_error = [loc_error[i] for i in inds]
area = [area[i] for i in inds]
bbox_w = [bbox_w[i] for i in inds]
bbox_h = [bbox_h[i] for i in inds]

inds = np.argsort(area, kind='mergesort')
area1 = [area[i] for i in inds]
loc_error1 = [loc_error[i] for i in inds]

fig = plt.figure()
plt.subplot(311)
plt.plot(area1, loc_error1)
plt.title('area vs localization error of {} score{}'.format(dataset, score_thr))
plt.xlabel('area')
plt.ylabel('localization error')
# plt.show()

inds = np.argsort(bbox_w, kind='mergesort')
bbox_w2 = [bbox_w[i] for i in inds]
loc_error2 = [loc_error[i] for i in inds]

# plt.figure()
plt.subplot(312)
plt.plot(bbox_w2, loc_error2)
plt.title('bbox width vs localization error of {} score{}'.format(dataset, score_thr))
plt.xlabel('bbox width')
plt.ylabel('localization error')
# plt.show()

inds = np.argsort(bbox_h, kind='mergesort')
bbox_h3= [bbox_h[i] for i in inds]
loc_error3 = [loc_error[i] for i in inds]

# plt.figure()
plt.subplot(313)
plt.plot(bbox_h3, loc_error3)
plt.title('bbox height vs localization error of {} score{}'.format(dataset, score_thr))
plt.xlabel('bbox height')
plt.ylabel('localization error')
plt.show()
fig.savefig('{}/localization_error_y_{}_scoreThr{}_smooth_L1.png'.format(output_dir, dataset, score_thr))
