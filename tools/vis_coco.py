from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Polygon
# import numpy as np
import json
import os

FOLDER_MAP = {
    'train': '/mnt/fcav/self_training/object_detection/dataset/cityscapes/annotations/instances_caronly_train.json',
    'val': '/mnt/fcav/self_training/object_detection/dataset/cityscapes/annotations/instances_caronly_val.json',
}

OUTPUT_MAP = {
    'train': '/mnt/fcav/self_training/object_detection/dataset/cityscapes/vis/train',
    'val': '/mnt/fcav/self_training/object_detection/dataset/cityscapes/vis/val',
}

data_dir = '/mnt/fcav/self_training/object_detection/dataset/cityscapes/images'

for data_set in ['train', 'val']:
    with open(FOLDER_MAP[data_set ]) as f:
        gt = json.load(f)
    coco = COCO(FOLDER_MAP[data_set])
    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds()
    imgIds = []
    for i in catIds:
        imgIds = coco.getImgIds(catIds=i) + imgIds

    imgIds = set(imgIds)
    image_num = 0
    for i in imgIds:
        img = coco.loadImgs(i)[0]
        print('>> processing number {}/{} image:{}'.format(image_num, len(imgIds), img['file_name']))
        image_num = image_num + 1
        I = io.imread('%s/%s' % (data_dir, img['file_name']))
        fig = plt.figure(figsize=(img['width']/100, img['height']/100))

        # fig.set_size_inches(img['width'], img['height'])
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(I)
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            bbox = ann['bbox']
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2],
                              bbox[3],
                              fill=False, edgecolor='g',
                              linewidth=2, alpha=0.4))
            ax.text(
                bbox[0], bbox[1] - 2,
                coco.loadCats(ann['category_id'])[0]['name'],
                fontsize=10,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')
        # plt.show()

        # plt.axis('off')
        # plt.imshow(I)
        # annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        # anns = coco.loadAnns(annIds)
        # coco.showAnns(anns)
        # plt.show()
        if not os.path.exists(OUTPUT_MAP[data_set]):
            os.makedirs(OUTPUT_MAP[data_set])
        fig.savefig(os.path.join(OUTPUT_MAP[data_set], '{}'.format(img['file_name'])), dpi=100)
        plt.close('all')