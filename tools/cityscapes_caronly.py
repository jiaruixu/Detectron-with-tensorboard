import os
from pycocotools.coco import COCO
import json

FOLDER_MAP = {
    'train': '/mnt/fcav/self_training/object_detection/dataset/cityscapes/annotations/instancesonly_filtered_gtFine_train.json',
    'val': '/mnt/fcav/self_training/object_detection/dataset/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
}

for data_set in ['train', 'val']:
    with open(FOLDER_MAP[data_set]) as f:
        gt = json.load(f)

    print('>> processing dataset {}'.format(data_set))

    coco = COCO(FOLDER_MAP[data_set ])

    img_list = []
    coco_new = dict()
    coco_new['images'] = []
    coco_new['annotations'] = gt['annotations']
    coco_new['categories'] = []

    coco_new['categories'].append(gt['categories'][0])

    catIds = coco.getCatIds()
    imgIds = []
    for i in catIds:
        imgIds = coco.getImgIds(catIds=i) + imgIds
    imgIds = set(imgIds)
    for i in imgIds:
        img = coco.loadImgs(i)[0]
        coco_new['images'].append(img)

    print ('>> {} set has {} images'.format(data_set, len(coco_new['images'])))

    for ann in coco_new['annotations']:
        ann['category_id'] = 1

    json_file = '/mnt/fcav/self_training/object_detection/dataset/cityscapes/annotations/instances_caronly_%s.json' % data_set
    json.dump(coco_new, open(json_file, 'w'))


