import argparse
import json
from pycocotools.coco import COCO
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--detections',
        dest='detections',
        help='detections json file',
        default='/mnt/fcav/self_training/object_detection/lowerbound/prediction/test/coco_KITTI_caronly_val/generalized_rcnn/bbox_coco_KITTI_caronly_val_results.json',
        type=str
    )
    parser.add_argument(
        '--ground-truth',
        dest='ground_truth',
        help='ground truth json file',
        default='/mnt/fcav/self_training/object_detection/dataset/KITTI/annotations/instances_caronly_val.json',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='output directory',
        default='/mnt/fcav/self_training/object_detection/lowerbound/prediction/test/coco_KITTI_caronly_val/predictions_txt',
        type=str
    )
    args = parser.parse_args()
    return args

def main(args):
    with open(args.detections, 'r') as f:
        dets = json.load(f)

    coco = COCO(args.ground_truth)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    imgs = coco.loadImgs(list(coco.imgs.keys()))
    for k, img in enumerate(imgs):
        print('[%6d/%6d] %s' % (k, len(imgs), img['file_name']))
        image_name, image_ext = os.path.splitext(img['file_name'])
        result_file_car_only = open(
            os.path.join(args.output_dir,
                         img['file_name'].replace(image_ext, '.txt')),
            'w')

        for ind, ann in enumerate(dets):
            if ann['image_id'] == img['id']:
                bbox = ann['bbox']
                score = ann['score']
                label = 'Car'
                result_file_car_only.write(
                    "%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.3f\n"
                    % (label, bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], score))



if __name__ == '__main__':
    args = parse_args()
    main(args)