from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import argparse
import sys
from copy import deepcopy

annotation_id = 0
category_item_id = 0
category_set = dict()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prediction-dir',
        dest='prediction_dir',
        help='prediction directory',
        default='/mnt/fcav/self_training/object_detection/lowerbound/prediction/test/coco_KITTI_tracking_imagesonly/generalized_rcnn/detections.pkl',
        type=str
    )
    parser.add_argument(
        '--image-coco-dir',
        dest='image_coco_dir',
        help='ground truth directory',
        default='/mnt/fcav/self_training/object_detection/dataset/KITTI_tracking/coco_KITTI_tracking_imagesonly.json',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='output directory',
        default='/mnt/fcav/self_training/object_detection/dataset/KITTI_tracking/annotations',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='detection prob threshold',
        default=0.999,
        type=float
    )
    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()
    return args

def addAnnoItem(coco, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)

def addCatItem(coco, name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id

def main(args):
    # read json files
    print('>>read images from {}'.format(args.image_coco_dir))
    with open(args.image_coco_dir) as f:
        image_coco = json.load(f)

    print('>>read predictions from {}'.format(args.prediction_dir))
    with open(args.prediction_dir) as f:
        prediction = json.load(f)
    with open(args.prediction_dir, 'r') as f:
        prediction = pickle.load(f)
    # initialization
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []

    # add coco['categories']
    current_category_id = addCatItem(coco, 'car')
    print('add category: car, id: {}'.format(current_category_id))
    # coco['categories'] = image_coco['categories']

    # add coco['annotations']
    pred_set = set([])
    for ind, annotation in enumerate(prediction):
        # add annotations that have score >= threshold
        if annotation['score'] >= args.thresh:
            print('>>processing predictions of image id {}'.format(annotation['image_id']))
            addAnnoItem(coco, annotation['image_id'], annotation['category_id'], annotation['bbox'])
            pred_set.add(annotation['image_id'])

    # add coco['images']
    for ind, img in enumerate(image_coco['images']):
        # add images that have annotations
        if img['id'] in pred_set:
            print('>>processing image id {}'.format(img['id']))
            imgDict = deepcopy(img)
            imgDict.pop('seg_file_name')
            coco['images'].append(imgDict)

    print('>>total number of images: {}'.format(len(pred_set)))

    json_file = '{}/instances_caronly_train_with_prediction.json'.format(args.output_dir)
    print('>>write to file: {}'.format(json_file))
    json.dump(coco, open(json_file, 'w'))

if __name__ == '__main__':
    args = parse_args()
    main(args)