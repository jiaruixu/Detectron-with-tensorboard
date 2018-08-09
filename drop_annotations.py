from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import argparse
import sys
import os

from random import shuffle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--annotation-dir',
        dest='annotation_dir',
        help='annotation directory',
        default='/mnt/fcav/self_training/object_detection/dataset/COCO/annotations/instances_val2014.json',
        type=str
    )
    parser.add_argument(
        '--drop-rate',
        dest='drop_rate',
        help='percentage of annotations to drop',
        default=0.3,
        type=float
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='output directory',
        default='',
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def main(args):
    # read json files
    print('>> Reading annotations from {}'.format(args.annotation_dir))
    with open(args.annotation_dir) as f:
        coco = json.load(f)

    # dict{image: number of annotations} count how many number of annotations in each image
    # dict{category id: annotations, number of total annotations}
    # for each category drop drop_rate% annotations
    # for each category:
    #   random shuffle annotations
    #   drop_num = drop_rate * annotations_num
    #   while(drop_num != 0):
    #       get the annotation to drop (drop annotations in order)
    #       check image id of the annoation to drop
    #       if image_num_annotations_dict[image_id] == 1:
    #           drop next annotation instead of this one
    #       else:
    #           drop this annotation
    #       drop_num--

    # initialization
    # dict{image: number of annotations} count how many number of annotations in each image
    # dict{category id: annotations, number of total annotations}
    print('>> Initialization...')
    image_id_list = [img['id'] for ind, img in enumerate(coco['images'])]  # num = 40504
    image_dict = dict()
    for image_id in image_id_list:
        image_dict[image_id] = 0

    cat_id_list = [cat['id'] for ind, cat in enumerate(coco['categories'])]
    cat_dict = dict()
    for cat_id in cat_id_list:
        cat_dict[cat_id] = dict()
        cat_dict[cat_id]['annotations'] = []
        cat_dict[cat_id]['annotations_num'] = 0

    print('>> Generating image dict and category dict...')
    for ind, ann in enumerate(coco['annotations']):
        image_id = ann['image_id']
        image_dict[image_id] += 1

        cat_id = ann['category_id']
        cat_dict[cat_id]['annotations'].append(ann)
        cat_dict[cat_id]['annotations_num'] += 1

    print('>> Finding the annotations to drop...')
    ann_drop_id = []

    for cat_id in cat_id_list:
        annotations = cat_dict[cat_id]['annotations']
        annotations_num = cat_dict[cat_id]['annotations_num']

        drop_num = int(round(args.drop_rate * annotations_num))
        random_list = range(annotations_num)
        shuffle(random_list)
        idx = 0
        while drop_num != 0:
            annotation_to_drop = annotations[random_list[idx]]
            image_id = annotation_to_drop['image_id']
            if image_dict[image_id] > 1:
                ann_drop_id.append(annotation_to_drop['id'])
                image_dict[image_id] -= 1
                drop_num -= 1
            idx += 1

    # drop annotations
    print('>> Dropping annotations...')
    for ind, ann in enumerate(coco['annotations']):
        if ann['id'] in ann_drop_id:
            coco['annotations'].pop(ind)

    name, _ = os.path.splitext(os.path.basename(args.annotation_dir))
    json_file = '{}/{}_droprate{}.json'.format(args.output_dir, name, args.drop_rate)
    print('>> Writing to file: {}'.format(json_file))
    json.dump(coco, open(json_file, 'w'))

if __name__ == '__main__':
    args = parse_args()
    main(args)