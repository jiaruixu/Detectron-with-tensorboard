import os
import glob

_FOLDERS_MAP = {
    'prediction': '/mnt/fcav/self_training/object_detection/lowerbound/prediction/test/coco_KITTI_caronly_val/predictions_txt',
    'gt': '/mnt/fcav/self_training/object_detection/dataset/KITTI/label_2',
}

_OUTPUT_MAP = {
    'prediction': '/mnt/data/KITTI/results/results/data',
    'gt': '/mnt/data/KITTI/data/object/label_2',
}

def _get_files(data):
    pattern = '*.txt'
    search_files = os.path.join(_FOLDERS_MAP[data], pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)

def main():
    prediction_files = _get_files('prediction') # prediction
    # gt_files = _get_files('gt') # ground truth
    num_files = len(prediction_files)

    for i in range(num_files):
        pre_name, _ = os.path.splitext(os.path.basename(prediction_files[i]))
        search_files = os.path.join(_FOLDERS_MAP['gt'], os.path.basename(prediction_files[i]))
        gt_files = glob.glob(search_files)
        gt_name, _ = os.path.splitext(os.path.basename(gt_files[0]))
        print(">>precessing image %s" % (gt_name))
        if gt_name != pre_name:
            print(">>ground truth %s is not correspond to %s" % (gt_name, pre_name))
            break

        new_pattern = '%06d.txt' % i

        if not os.path.exists(_OUTPUT_MAP['gt']):
            os.makedirs(_OUTPUT_MAP['gt'])
        gt_new_file = os.path.join(_OUTPUT_MAP['gt'], new_pattern)
        gt = open(gt_files[0])
        gt_new = open(gt_new_file, "w")
        for line in gt.readlines():
            line_sp = line.split()
            if line_sp[0] == 'Truck' or line_sp[0] == 'Van':
                line = line.replace(line_sp[0], 'Car')
            gt_new.write(line)

        if not os.path.exists(_OUTPUT_MAP['prediction']):
            os.makedirs(_OUTPUT_MAP['prediction'])
        pre_new_file = os.path.join(_OUTPUT_MAP['prediction'], new_pattern)
        pre = open(prediction_files[i])
        pre_new = open(pre_new_file, "w")
        for line in pre.readlines():
            pre_new.write(line)

if __name__ == '__main__':
    main()
