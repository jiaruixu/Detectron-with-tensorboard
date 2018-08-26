import pandas as pd
import skimage.io as io
import os
import matplotlib.pyplot as plt

file = '/mnt/fcav/self_training/object_detection/lowerbound/prediction/test/cityscapes_caronly_val/generalized_rcnn/ious_checkpoint16999.xlsx'
output_dir = '/mnt/fcav/self_training/object_detection/lowerbound/prediction/test/cityscapes_caronly_val/localization_error'
xl = pd.ExcelFile(file)
df = xl.parse('Sheet1')
row = df.values
row0 = row[3054]

# frankfurt_000001_054077_leftImg8bit.png

data_dir = '/mnt/fcav/self_training/object_detection/dataset/cityscapes/images'
file_name = row0[1]
I = io.imread('%s/%s' % (data_dir, file_name))
fig = plt.figure(figsize=(2048/100, 1024/100))

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.axis('off')
fig.add_axes(ax)
ax.imshow(I)

dt_bbox = [row0[2], row0[3], row0[4], row0[5]]
gt_bbox = [row0[9], row0[10], row0[11], row0[12]]
ax.add_patch(
        plt.Rectangle((dt_bbox[0], dt_bbox[1]),
                       dt_bbox[2],
                       dt_bbox[3],
                       fill=False, edgecolor='g',
                       linewidth=2, alpha=0.4))
ax.add_patch(
        plt.Rectangle((gt_bbox[0], gt_bbox[1]),
                       gt_bbox[2],
                       gt_bbox[3],
                       fill=False, edgecolor='r',
                       linewidth=2, alpha=0.4))
plt.show()
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
fig.savefig(os.path.join(output_dir, '{}'.format(file_name)), dpi=100)
