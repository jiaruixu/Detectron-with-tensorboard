from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import argparse
import glob
import sys
import os
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
        default='/mnt/ngv/datasets/KITTI/training',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='output directory',
        default='/mnt/fcav/self_training/object_detection/dataset/KITTI/annotations',
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

def addImgItem(file_name, size):
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
    search_files = os.path.join(args.data_dir, 'label_2', '*.txt')
    label_filenames = glob.glob(search_files)
    label_filenames = sorted(label_filenames)
    current_image_id = None

    current_category_id = addCatItem('car')
    print('add annotation: Car')

    for file in label_filenames:
        f = open(file, 'r')
        filename = os.path.basename(file)
        ann_num = 0  # only add image item when first add annotation in this image
        for line in f:
            one_line = line.strip()
            words = one_line.split()
            if words[0] == 'Truck' or words[0] == 'Car' or words[0] == 'Van':
                ann_num += 1
                if ann_num == 1:
                    search_image_files = os.path.join(args.data_dir, 'image_2', filename.replace('.txt', '.png'))
                    image_file = sorted(glob.glob(search_image_files))
                    image = img.open(image_file[0])
                    image_size = dict()
                    image_size['width'] = None
                    image_size['height'] = None
                    image_size['depth'] = None
                    image_size['width'], image_size['height'] = image.size
                    # image_name, _ = os.path.splitext(os.path.basename(image_file[0]))
                    current_image_id = addImgItem(os.path.basename(image_file[0]), image_size)
                    print('add image with {} and {}'.format(os.path.basename(image_file[0]), image_size))

                bbox = []
                bbox.append(float(words[4]))
                bbox.append(float(words[5]))
                bbox.append(round(float(words[6]) - float(words[4]), 2))
                bbox.append(round(float(words[7]) - float(words[5]), 2))
                print('add annotation with {},{},{},{}'.format('car', current_image_id, current_category_id, bbox))
                addAnnoItem('car', current_image_id, current_category_id, bbox)

        # if current_image_id == 1000:
        #    break

if __name__ == '__main__':
    args = parse_args()
    main(args)
    json_file = os.path.join(args.output_dir, 'instances_caronly_val_2.json')
    json.dump(coco, open(json_file, 'w'))