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
        '--json-dir',
        dest='json_dir',
        help='json directory',
        default='',
        type=str
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

def sortMethod(img):
    return img['id']

def main(args):
    # read json files
    print('>> Reading json file from {}'.format(args.json_dir))
    with open(args.json_dir) as f:
        data = json.load(f)

    if os.path.basename(args.json_dir).find('train') != -1:
        txt_name = 'train.txt'
    else:
        txt_name = 'val.txt'

    print('>> Writing to txt...')
    txt_path = os.path.join(args.output_dir, txt_name)
    f_txt = open(txt_path, 'w')

    images = data['images']
    images.sort(key=sortMethod)
    # image_id_list = [img['id'] for idx, img in enumerate(images)]
    # image_id_list.sort()

    # for id in image_id_list:
    for img in images:
        file_name = img['file_name']
        name, _ = os.path.splitext(file_name)
        print('>> Writing image: {} image_id: {} to txt as: {} '.format(file_name, img['id'], name))
        f_txt.write(name + '\n')

    f_txt.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)