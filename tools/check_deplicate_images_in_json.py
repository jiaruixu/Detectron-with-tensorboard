import json

gt_dir = '/mnt/fcav/self_training/object_detection/dataset/cityscapes/annotations/instancesonly_filtered_gtFine_train.json'
with open(gt_dir) as f:
    gt = json.load(f)

gt_dir2 = '/mnt/fcav/self_training/object_detection/dataset/cityscapes/annotations/instancesonly_filtered_gtFine_val.json'
with open(gt_dir2) as f:
    gt2 = json.load(f)

image_set = set()

img1 = gt['images']
img2 = gt2['images']
for ind, img in enumerate(img1):
    file_name = img['file_name']
    # if file_name.find('2979535.jpg') != -1:
    #    raise ValueError('find 2979535.jpg')
    if file_name not in image_set:
        image_set.add(file_name)
        print(' {}: add train image with {}'.format(ind, file_name))
    else:
        raise Exception('{}: duplicated image: {}'.format(ind, file_name))

for ind, img2 in enumerate(img2):
    file_name = img2['file_name']
    # if file_name.find('2979535.jpg') != -1:
    #    raise ValueError('find 2979535.jpg')
    if file_name not in image_set:
        image_set.add(file_name)
        print(' {}: add train image with {}'.format(ind, file_name))
    else:
        raise Exception('{}: duplicated image: {}'.format(ind, file_name))