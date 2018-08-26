from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import argparse
import glob
import sys
import os
import cPickle as pickle
import PIL.Image as img

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

category_item_id = 0
image_id = 0
annotation_id = 0

def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--data-dir',
        dest='data_dir',
        help='data directory',
        default='/mnt/ngv/self-supervised-learning/Datasets/KITTI/tracking/testing/image_02',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='output directory',
        default='/mnt/fcav/self_training/object_detection/dataset/KITTI_tracking',
        type=str
    )
    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    return parser.parse_args()

def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id

def addImgItem(file_name, img_series, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_item = dict()
    image_id += 1
    image_item['id'] = image_id
    image_item['series'] = img_series
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id

def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    #bbox[] is x,y,w,h
    #left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    #left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    #right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    #right_top
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
    search_files = os.path.join(args.data_dir, '*', '*.png')
    image_filenames = glob.glob(search_files)
    image_filenames = sorted(image_filenames)

    addCatItem('car')

    for image_file in image_filenames:
        image = img.open(image_file)
        img_series = os.path.split(os.path.dirname(image_file))[1]
        image_size = dict()
        image_size['width'] = None
        image_size['height'] = None
        image_size['depth'] = None
        image_size['width'], image_size['height'] = image.size
        image_name = '%s_%s' % (img_series, os.path.basename(image_file))
        current_image_id = addImgItem(image_name, img_series, image_size)
        print('add image with {} and {}'.format(image_name, image_size))
        # save_image = os.path.join(args.output_dir, 'image_2', image_name)
        # save_image = "/mnt/fcav/self_training/object_detection/dataset/KITTI_tracking/image_2/{}".format(image_name)
        # image.save(save_image)
        # if current_image_id == 50:
        #    break

if __name__ == '__main__':
    args = parse_args()
    main(args)
    json_file = os.path.join(args.output_dir, 'KITTI_tracking_imagesonly.json')
    json.dump(coco, open(json_file, 'w'))