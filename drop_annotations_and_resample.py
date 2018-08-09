from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import argparse
import sys
import os

from random import shuffle, sample

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
        '--sample-number',
        dest='sample_number',
        help='number of images to sample',
        default=20000,
        type=int
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
        data = json.load(f)

    # image_dict{image id: number of annotations} count how many number of annotations in each image
    # cat_dict{category id: annotations, number of total annotations}
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
    # image_dict{image id: number of annotations} count how many number of annotations in each image
    # cat_dict{category id: annotations, number of total annotations}
    print('>> Initialization...')

    coco = dict()
    coco['images'] = []
    coco['categories'] = data['categories']
    coco['annotations'] = []
    coco['info'] = data['info']
    coco['licenses'] = data['licenses']

    coco_nondrop = dict()
    coco_nondrop['images'] = []
    coco_nondrop['categories'] = data['categories']
    coco_nondrop['annotations'] = []
    coco_nondrop['info'] = data['info']
    coco_nondrop['licenses'] = data['licenses']

    print('>> Random sampling and initialize image dict...')
    # Random sample 20k images
    random_list = sample(xrange(len(data['images'])), args.sample_number)
    image_id_list = []
    image_dict = dict()
    for ind in random_list:
        img = data['images'][ind]
        coco['images'].append(img)
        image_id_list.append(img['id'])
        image_dict[img['id']] = 0

    coco_nondrop['images'] = coco['images']

    print('>> Initialize category dict...')
    cat_id_list = [cat['id'] for ind, cat in enumerate(data['categories'])]
    cat_dict = dict()
    for cat_id in cat_id_list:
        cat_dict[cat_id] = dict()
        cat_dict[cat_id]['annotations'] = []
        cat_dict[cat_id]['annotations_num'] = 0

    print('>> Generating image dict and category dict...')
    sampled_annotations = []
    sampled_dataset_cat = []
    ann_id_list = []
    for ind, ann in enumerate(data['annotations']):
        image_id = ann['image_id']
        if image_id in image_id_list:
            image_dict[image_id] += 1

            sampled_annotations.append(ann)
            ann_id_list.append(ann['id'])

            cat_id = ann['category_id']
            sampled_dataset_cat.append(cat_id)
            cat_dict[cat_id]['annotations'].append(ann)
            cat_dict[cat_id]['annotations_num'] += 1

    coco_nondrop['annotations'] = sampled_annotations

    # check if any categories are dropped during sampling
    sampled_dataset_cat = set(sampled_dataset_cat)
    missing_cat = list(set(cat_id_list).difference(sampled_dataset_cat))
    if len(missing_cat) != 0:
        print('>> Categoties {} is missing...'.format(missing_cat))

    print('>> Finding annotations to drop...')
    ann_drop_id = []
    for cat_id in sampled_dataset_cat:
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
    ann_keep_id = list(set(ann_id_list).difference(set(ann_drop_id)))

    for ind, ann in enumerate(sampled_annotations):
        if ann['id'] in ann_keep_id:
            coco['annotations'].append(ann)

    name, _ = os.path.splitext(os.path.basename(args.annotation_dir))
    json_file_drop = '{}/{}_droprate{}_sample{}.json'.format(args.output_dir, name, args.drop_rate, args.sample_number)
    print('>> Writing to file: {}'.format(json_file_drop))
    json.dump(coco, open(json_file_drop, 'w'))

    json_file_nondrop = '{}/{}_sample{}.json'.format(args.output_dir, name, args.sample_number)
    print('>> Writing to file: {}'.format(json_file_nondrop))
    json.dump(coco_nondrop, open(json_file_nondrop, 'w'))

if __name__ == '__main__':
    args = parse_args()
    main(args)