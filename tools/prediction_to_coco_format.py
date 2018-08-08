from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import argparse
import sys
from copy import deepcopy

annotation_id = 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prediction-dir',
        dest='prediction_dir',
        help='prediction directory',
        default='/mnt/fcav/self_training/object_detection/lowerbound/prediction_on_cityscapes_train/bbox_cityscapes_caronly_train_results.json',
        type=str
    )
    parser.add_argument(
        '--gt-dir',
        dest='gt_dir',
        help='ground truth directory',
        default='/mnt/fcav/self_training/object_detection/dataset/cityscapes/annotations/instancesonly_filtered_gtFine_train.json',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='output directory',
        default='/mnt/fcav/self_training/object_detection/lowerbound/prediction_on_cityscapes_train',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='detection prob threshold',
        default=0.96,
        type=float
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
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

def main(args):
    # read json files
    print('>>read ground truth from {}'.format(args.gt_dir))
    with open(args.gt_dir) as f:
        gt = json.load(f)

    print('>>read predictions from {}'.format(args.prediction_dir))
    with open(args.prediction_dir) as f:
        prediction = json.load(f)

    # initialization
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []

    # add coco['categories']
    coco['categories'] = gt['categories']

    # add coco['annotations']
    pred_set = set([])
    for ind, annotation in enumerate(prediction):
        # add annotations that have score >= threshold
        if annotation['score'] >= args.thresh:
            print('>>processing predictions of image id {}'.format(annotation['image_id']))
            addAnnoItem(coco, annotation['image_id'], annotation['category_id'], annotation['bbox'])
            pred_set.add(annotation['image_id'])

    # add coco['images']
    for ind, img in enumerate(gt['images']):
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