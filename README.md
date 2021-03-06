## Detectron with tensorboard
This repository was forked from [Detectron with tensorboard](https://github.com/tfzhou/Detectron-with-tensorboard)

`tools/pascal_voc_xml2json.py` was forked from [Gemfield](https://github.com/CivilNet/Gemfield/blob/master/src/python/pascal_voc_xml2json/pascal_voc_xml2json.py)

This repository use [c2board](https://github.com/endernewton/c2board) to visualize some training info of Detectron in tensorboard. It dumps the training info in the output folder of Detectron by default.

## Dataset preparation
### Transfer GTA and Cityscapes to COCO format

1. Extract only car annotations of GTA and Cityscapes.
2. Convert annotations to COCO format.

### Cityscapes to COCO Information

```
python2 tools/convert_cityscapes_to_coco.py \
	--dataset cityscapes_instance_only \
	--outdir /mnt/fcav/self_training/object_detection/dataset/cityscapes/annotations \
	--datadir /mnt/fcav/continuous_learning/datasets/cityscapes/raw
```

### Sample GTA dataset

Extract 8000 images with car annoations as train set and 2000 images with car annotations as valiation set.

Not all the images in the GTA dataset has car annoations. So we first extract images with car annotations from GTA 200k dataset (instead of GTA 10k dataset) and then generate the train and val set by random sampling.

Generate GTA train set (8000 images)

```
python2 tools/drop_annotations_and_resample.py \
	--annotation-dir ./detectron/datasets/data/GTA_Pascal_format/Annotations/instances_caronly_train.json \
	--sample-number 8000 \
	--sample-only True \
	--output-dir ./detectron/datasets/data/GTA_Pascal_format/Annotations
```

Generate GTA val set (2000 images)

```
python2 tools/drop_annotations_and_resample.py \
	--annotation-dir ./detectron/datasets/data/GTA_Pascal_format/Annotations/instances_caronly_val.json \
	--sample-number 2000 \
	--sample-only True \
	--output-dir ./detectron/datasets/data/GTA_Pascal_format/Annotations
```

Write image names to txt

```
# train
python tools/write_name_to_txt.py \
	--json-dir /mnt/fcav/self_training/object_detection/dataset/GTA_Pascal_format/Annotations/instances_caronly_train_sample8000.json \
	--output-dir /mnt/fcav/self_training/object_detection/dataset/GTA_Pascal_format/VOCdevkit2012/VOC2012/ImageSets/Main

# val
python tools/write_name_to_txt.py \
	--json-dir /mnt/fcav/self_training/object_detection/dataset/GTA_Pascal_format/Annotations/instances_caronly_val_sample2000.json \
	--output-dir /mnt/fcav/self_training/object_detection/dataset/GTA_Pascal_format/VOCdevkit2012/VOC2012/ImageSets/Main
```

### Add dataset to catalog and modify configs

`./detectron/datasets/dataset_catalog.py`

`./configs/configs/`

## Train Faster RCNN

Use `DetectronDocker` repository

## Visualize average precision

`eval.py` will plot the **precision-recall curve** and save the **corresponding scores** into an excel. While evaluating, you can also use tensorboard to visualize the change of average precision. It dumps the info in the output folder of Detectron by default.

### visualize lower bound
visualize lower bound on GTA

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound/train/voc_GTA_caronly_train_sample8000/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/lowerbound/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/lowerbound/eval/e2e-eval-logs.txt
```
```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound_discard/train/voc_GTA_caronly_train_and_voc_GTA_caronly_val/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/lowerbound_discard//eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/lowerbound_discard//eval/e2e-eval-logs.txt
```
visualize lower bound on Cityscapes

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_eval.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound/train/voc_GTA_caronly_train_sample8000/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/lowerbound/eval
```

visualize lower bound on KITTI

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_evalKITTI.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound_more_checkpoints/train/voc_GTA_caronly_train_sample8000/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/lowerbound_more_checkpoints/eval
```

### visualize upperbound1

visualize upperbound1 on GTA

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_upperbound1.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/upperbound1/train/voc_GTA_caronly_train_sample8000:cityscapes_caronly_train/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/upperbound1/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/upperbound1/eval/e2e-eval-logs.txt
```

visualize upperbound1 on KITTI

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_upperbound1_eval.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/upperbound1/train/voc_GTA_caronly_train_sample8000:coco_KITTI_caronly_train/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/upperbound1/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/upperbound1/eval/e2e-eval-KITTI-logs.txt
```

### visualize baseline

on GTA

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/baseline_basenetwork_GTA8k/baseline/train/voc_GTA_caronly_train_sample8000:coco_KITTI_tracking_train_with_prediction/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/baseline_basenetwork_GTA8k/baseline/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/baseline_basenetwork_GTA8k/baseline/e2e-eval-GTA-logs.txt
```

on KITTI eval

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_evalKITTI1000.yaml \
	--start-checkpoint 26000 \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/baseline_basenetwork_GTA8k/baseline/train/voc_GTA_caronly_train_sample8000:coco_KITTI_tracking_train_with_prediction/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/baseline_basenetwork_GTA8k/baseline/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/baseline_basenetwork_GTA8k/baseline/e2e-eval-KITTI-logs.txt
```

### visualize baseline_ss

on GTA

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/baseline_ss/train/voc_GTA_caronly_train_sample8000:coco_KITTI_tracking_train_with_prediction/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/baseline_ss/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/baseline_ss/e2e-eval-GTA-logs.txt
```

on KITTI eval

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss_evalKITTI1000.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/baseline_ss/train/voc_GTA_caronly_train_sample8000:coco_KITTI_tracking_train_with_prediction/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/baseline_ss/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/baseline_ss/e2e-eval-KITTI-logs.txt
```
### visualize localization error test

localization_error_test

on GTA

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_localization_error_test.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/localization_error_test/train/voc_GTA_caronly_train_sample8000:coco_KITTI_caronly_tp_preds/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/localization_error_test/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/localization_error_test/e2e-eval-GTA-logs.txt
```

on KITTI eval

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_localization_error_test_evalKITTI1k.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/localization_error_test/train/voc_GTA_caronly_train_sample8000:coco_KITTI_caronly_tp_preds/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/localization_error_test/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/localization_error_test/e2e-eval-KITTI-logs.txt
```

localization_error_test_no_fn

on GTA

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_localization_error_test_no_fn.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/localization_error_test_no_fn/train/voc_GTA_caronly_train_sample8000:coco_KITTI_caronly_tp_preds/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/localization_error_test_no_fn/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/localization_error_test_no_fn/e2e-eval-GTA-logs.txt
```

on KITTI eval

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_localization_error_test_no_fn_evalKITTI1k.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/localization_error_test_no_fn/train/voc_GTA_caronly_train_sample8000:coco_KITTI_caronly_tp_preds_no_fn/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/localization_error_test_no_fn/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/localization_error_test_no_fn/e2e-eval-KITTI-logs.txt
```

localization_error_test_no_fn_ss_with_10fp

on KITTI eval

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_localization_error_test_no_fn_with_10fp.yaml \
	TEST.WEIGHTS  \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/localization_error_test_no_fn_ss_with_10fp/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/localization_error_test_no_fn_ss_with_10fp/e2e-eval-KITTI-logs.txt
```

localization_error_test_no_fn_ss_with_20fp

on KITTI eval

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_localization_error_test_no_fn_with_20fp.yaml \
	TEST.WEIGHTS  \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/localization_error_test_no_fn_ss_with_20fp/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/localization_error_test_no_fn_ss_with_20fp/e2e-eval-KITTI-logs.txt
```

localization_error_test_no_fn_ss

on KITTI eval

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_localization_error_test_no_fn_evalKITTI1k.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/localization_error_test_no_fn_ss/train/voc_GTA_caronly_train_sample8000:coco_KITTI_caronly_tp_preds_no_fn/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/localization_error_test_no_fn_ss/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/localization_error_test_no_fn_ss/e2e-eval-KITTI-logs.txt
```

### visualize baseline_ss_GTA8k_pred

on KITTI eval

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss_GTA8k_pred_evalKITTIval1k.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/baseline_ss_retrain_GTA8k_pred/train/voc_GTA_caronly_train_sample8000:coco_KITTI_train_with_prediction/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/baseline_ss_retrain_GTA8k_pred/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/baseline_ss_retrain_GTA8k_pred/e2e-eval-KITTI-logs.txt
```
### visualize baseline_ss_trainconv

on GTA

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss_trainconv.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/baseline_ss_trainconv/train/voc_GTA_caronly_train_sample8000:coco_KITTI_tracking_train_with_prediction/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/baseline_ss_trainconv/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/baseline_ss_trainconv/e2e-eval-GTA-logs.txt
```

on KITTI eval

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss_evalKITTI1000.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/baseline_ss_trainconv/train/voc_GTA_caronly_train_sample8000:coco_KITTI_tracking_train_with_prediction/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/baseline_ss_trainconv/eval \
	2>&1 | tee -a /mnt/fcav/self_training/object_detection/baseline_ss_trainconv/e2e-eval-KITTI-logs.txt
```

### visualize upperbound2

```
python2 tools/eval.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_upperbound2.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/upperbound2/train/voc_GTA_caronly_train_sample8000:cityscapes_caronly_train_with_dropannotations/generalized_rcnn \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/upperbound2/eval
```

### Use tensorboard

```
tensorboard --logdir=$OUTPUT_DIR
```

## evaluate

lowerbound

on Cityscapes

```
python2 tools/test_net.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_eval.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound/train/voc_GTA_caronly_train_sample8000/generalized_rcnn/model_iter16999.pkl \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/lowerbound/prediction
```
on GTA

```
python2 tools/test_net.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound/train/voc_GTA_caronly_train_sample8000/generalized_rcnn/model_iter16999.pkl \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/lowerbound/prediction
```
on KITTI

```
python2 tools/test_net.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_evalKITTI.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound/train/voc_GTA_caronly_train_sample8000/generalized_rcnn/model_iter16999.pkl \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/lowerbound/prediction
```

on KITTI val 1000

```
python2 tools/test_net.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_evalKITTI1000.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train:voc_GTA_caronly_val/generalized_rcnn/model_final.pkl \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/prediction_final
```

upperbound1

```
python2 tools/test_net.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_upperbound1_eval.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/upperbound1/train/voc_GTA_caronly_train_sample8000:cityscapes_caronly_train/generalized_rcnn/model_iter69999.pkl \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/upperbound1/eval
```

## Prediction

# predict with lowerbound

on KITTI tracking dataset

```
python2 tools/test_net.py \
  --cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_prediction.yaml \
  TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound/train/voc_GTA_caronly_train_sample8000/generalized_rcnn/model_iter16999.pkl \
  NUM_GPUS 1 \
  OUTPUT_DIR /mnt/fcav/self_training/object_detection/lowerbound/prediction
```

on KITTI obj train

```
python2 tools/test_net.py \
  --cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_kitti_train.yaml \
  TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound/train/voc_GTA_caronly_train_sample8000/generalized_rcnn/model_iter16999.pkl \
  NUM_GPUS 1 \
  OUTPUT_DIR /mnt/fcav/self_training/object_detection/lowerbound/prediction
```

```
python2 tools/test_net.py \
  --cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_kitti_train.yaml \
  TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound_discard/train/voc_GTA_caronly_train:voc_GTA_caronly_val/generalized_rcnn/model_final.pkl \
  NUM_GPUS 1 \
  OUTPUT_DIR /mnt/fcav/self_training/object_detection/lowerbound/prediction
```

on cityscapes_caronly_val

```
python2 tools/test_net.py \
	--cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_eval.yaml \
	TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound/train/voc_GTA_caronly_train_sample8000/generalized_rcnn/model_iter16999.pkl \
	NUM_GPUS 1 \
	OUTPUT_DIR /mnt/fcav/self_training/object_detection/lowerbound/prediction
```
# predict with baseline_ss

on KITTI obj val

```
python2 tools/test_net.py \
  --cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_baseline_ss_GTA8k_pred_evalKITTIval1k.yaml \
  TEST.WEIGHTS /mnt/fcav/self_training/object_detection/baseline_ss_retrain_GTA8k_pred/train/voc_GTA_caronly_train_sample8000_coco_KITTI_train_with_prediction/generalized_rcnn/model_final.pkl \
  NUM_GPUS 1 \
  OUTPUT_DIR /mnt/fcav/self_training/object_detection/baseline_ss_retrain_GTA8k_pred/prediction
```

# visualize prediction

Cityscapes val

```
python2 tools/visualize_results.py \
	--dataset cityscapes_caronly_val \
	--detections /mnt/fcav/self_training/object_detection/lowerbound/prediction/test/cityscapes_caronly_val/generalized_rcnn/detections.pkl \
	--thresh 0.98 \
	--output-dir /mnt/fcav/self_training/object_detection/lowerbound/prediction/test/cityscapes_caronly_val
```

Cityscapes train

```
python2 tools/visualize_results.py \
	--dataset cityscapes_caronly_train \
	--detections /mnt/fcav/self_training/object_detection/lowerbound/prediction/test/cityscapes_caronly_train/generalized_rcnn/detections.pkl \
	--thresh 0.98 \
	--output-dir /mnt/fcav/self_training/object_detection/lowerbound/prediction/test/cityscapes_caronly_train
```

KITTI

```
python2 tools/visualize_results.py \
	--dataset coco_KITTI_caronly_val1000 \
	--detections /mnt/fcav/self_training/object_detection/lowerbound/prediction/test/coco_KITTI_caronly_val1000/generalized_rcnn/detections.pkl \
	--thresh 1 \
	--output-dir /mnt/fcav/self_training/object_detection/lowerbound/prediction/test/coco_KITTI_caronly_val1000
```

## Generate region proposals
GTA train 8k

```
python2 tools/test_net.py \
  --cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_GTA_train8k.yaml \
  TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train:voc_GTA_caronly_val/generalized_rcnn/model_final.pkl \
  NUM_GPUS 1 \
  OUTPUT_DIR /mnt/fcav/self_training/object_detection/region_proposals_GTA200k
```

GTA val 2k

```
python2 tools/test_net.py \
  --cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_GTA_val2k.yaml \
  TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train:voc_GTA_caronly_val/generalized_rcnn/model_final.pkl \
  NUM_GPUS 1 \
  OUTPUT_DIR /mnt/fcav/self_training/object_detection/region_proposals_GTA200k
```
GTA trainval

```
python2 tools/test_net.py \
  --cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_GTA_trainval.yaml \
  TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train:voc_GTA_caronly_val/generalized_rcnn/model_final.pkl \
  NUM_GPUS 1 \
  OUTPUT_DIR /mnt/fcav/self_training/object_detection/region_proposals_GTA200k
```

KITTI train with prediction

```
python2 tools/test_net.py \
  --cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_train_prediction.yaml \
  TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train:voc_GTA_caronly_val/generalized_rcnn/model_final.pkl \
  NUM_GPUS 1 \
  OUTPUT_DIR /mnt/fcav/self_training/object_detection/region_proposals_GTA200k
```

KITTI eval (object detection dataset)

```
python2 tools/test_net.py \
  --cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_eval.yaml \
  TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train:voc_GTA_caronly_val/generalized_rcnn/model_final.pkl \
  NUM_GPUS 1 \
  OUTPUT_DIR /mnt/fcav/self_training/object_detection/region_proposals_GTA200k
```

KITTI eval (object detection dataset) 1k

```
python2 tools/test_net.py \
  --cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_eval_1k.yaml \
  TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train:voc_GTA_caronly_val/generalized_rcnn/model_final.pkl \
  NUM_GPUS 1 \
  OUTPUT_DIR /mnt/fcav/self_training/object_detection/region_proposals_GTA200k
```

KITTI obj train

```
python2 tools/test_net.py \
  --cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_train.yaml \
  TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train:voc_GTA_caronly_val/generalized_rcnn/model_final.pkl \
  NUM_GPUS 1 \
  OUTPUT_DIR /mnt/fcav/self_training/object_detection/region_proposals_GTA200k
```

KITTI tp Predictions

```
python2 tools/test_net.py \
  --cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_tp_preds.yaml \
  TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train:voc_GTA_caronly_val/generalized_rcnn/model_final.pkl \
  NUM_GPUS 1 \
  OUTPUT_DIR /mnt/fcav/self_training/object_detection/region_proposals_GTA200k
```

KITTI tp Predictions no fn

```
python2 tools/test_net.py \
  --cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_tp_preds_no_fn.yaml \
  TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train:voc_GTA_caronly_val/generalized_rcnn/model_final.pkl \
  NUM_GPUS 1 \
  OUTPUT_DIR /mnt/fcav/self_training/object_detection/region_proposals_GTA200k
```

KITTI tp Predictions no fn with 10% fp (5605)

```
python2 tools/test_net.py \
  --cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_tp_preds_no_fn_with_10fp.yaml \
  TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train:voc_GTA_caronly_val/generalized_rcnn/model_final.pkl \
  NUM_GPUS 1 \
  OUTPUT_DIR /mnt/fcav/self_training/object_detection/region_proposals_GTA200k
```

KITTI tp Predictions no fn with 20% fp (5635)

```
python2 tools/test_net.py \
  --cfg /mnt/fcav/self_training/object_detection/configs/e2e_faster_rcnn_X-101-64x4d-FPN_1x_lowerbound_RPNONLY_KITTI_tp_preds_no_fn_with_20fp.yaml \
  TEST.WEIGHTS /mnt/fcav/self_training/object_detection/lowerbound_200k_trainval/train/voc_GTA_caronly_train:voc_GTA_caronly_val/generalized_rcnn/model_final.pkl \
  NUM_GPUS 1 \
  OUTPUT_DIR /mnt/fcav/self_training/object_detection/region_proposals_GTA200k
```

## Transfer prediction to coco format

Predictions are saved as list in json format. This scipt transfer predictions with score >= 0.96 to json format.

```
python2 tools/prediction_to_coco_format.py \
	--gt-dir /mnt/fcav/self_training/object_detection/dataset/cityscapes/annotations/instancesonly_filtered_gtFine_train.json \
	--prediction-dir /mnt/fcav/self_training/object_detection/lowerbound/prediction_on_cityscapes_train/bbox_cityscapes_caronly_train_results.json \
	--output-dir /mnt/fcav/self_training/object_detection/lowerbound/prediction_on_cityscapes_train \
	--thresh 0.96
```

## Drop Annotations

For dropping annotations only:

```
python2 drop_annotations.py \
	--annotation-dir /mnt/fcav/self_training/object_detection/dataset/COCO/annotations/instances_train2014.json \
	--drop-rate 0.3 \
	--output-dir /mnt/fcav/self_training/object_detection/dataset/COCO/annotations_drop
```

For COCO dataset, sampling certain number of images and dropping annotations:

```
python2 drop_annotations_and_resample.py \
	--annotation-dir /mnt/fcav/self_training/object_detection/dataset/COCO/annotations/instances_train2014.json \
	--drop-rate 0.3 \
	--sample-number 20000 \
	--output-dir /mnt/fcav/self_training/object_detection/dataset/COCO/annotations_drop
```

## Detectron

Detectron is Facebook AI Research's software system that implements state-of-the-art object detection algorithms, including [Mask R-CNN](https://arxiv.org/abs/1703.06870). It is written in Python and powered by the [Caffe2](https://github.com/caffe2/caffe2) deep learning framework.

At FAIR, Detectron has enabled numerous research projects, including: [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144), [Mask R-CNN](https://arxiv.org/abs/1703.06870), [Detecting and Recognizing Human-Object Interactions](https://arxiv.org/abs/1704.07333), [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002), [Non-local Neural Networks](https://arxiv.org/abs/1711.07971), [Learning to Segment Every Thing](https://arxiv.org/abs/1711.10370), [Data Distillation: Towards Omni-Supervised Learning](https://arxiv.org/abs/1712.04440), [DensePose: Dense Human Pose Estimation In The Wild](https://arxiv.org/abs/1802.00434), and [Group Normalization](https://arxiv.org/abs/1803.08494).

<div align="center">
  <img src="demo/output/33823288584_1d21cf0a26_k_example_output.jpg" width="700px" />
  <p>Example Mask R-CNN output.</p>
</div>

## Introduction

The goal of Detectron is to provide a high-quality, high-performance
codebase for object detection *research*. It is designed to be flexible in order
to support rapid implementation and evaluation of novel research. Detectron
includes implementations of the following object detection algorithms:

- [Mask R-CNN](https://arxiv.org/abs/1703.06870) -- *Marr Prize at ICCV 2017*
- [RetinaNet](https://arxiv.org/abs/1708.02002) -- *Best Student Paper Award at ICCV 2017*
- [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- [RPN](https://arxiv.org/abs/1506.01497)
- [Fast R-CNN](https://arxiv.org/abs/1504.08083)
- [R-FCN](https://arxiv.org/abs/1605.06409)

using the following backbone network architectures:

- [ResNeXt{50,101,152}](https://arxiv.org/abs/1611.05431)
- [ResNet{50,101,152}](https://arxiv.org/abs/1512.03385)
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144) (with ResNet/ResNeXt)
- [VGG16](https://arxiv.org/abs/1409.1556)

Additional backbone architectures may be easily implemented. For more details about these models, please see [References](#references) below.

## Update

- 4/2018: Support Group Normalization - see [`GN/README.md`](./projects/GN/README.md)

## License

Detectron is released under the [Apache 2.0 license](https://github.com/facebookresearch/detectron/blob/master/LICENSE). See the [NOTICE](https://github.com/facebookresearch/detectron/blob/master/NOTICE) file for additional details.

## Citing Detectron

If you use Detectron in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```
@misc{Detectron2018,
  author =       {Ross Girshick and Ilija Radosavovic and Georgia Gkioxari and
                  Piotr Doll\'{a}r and Kaiming He},
  title =        {Detectron},
  howpublished = {\url{https://github.com/facebookresearch/detectron}},
  year =         {2018}
}
```

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron Model Zoo](MODEL_ZOO.md).

## Installation

Please find installation instructions for Caffe2 and Detectron in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using Detectron

After installation, please see [`GETTING_STARTED.md`](GETTING_STARTED.md) for brief tutorials covering inference and training with Detectron.

## Getting Help

To start, please check the [troubleshooting](INSTALL.md#troubleshooting) section of our installation instructions as well as our [FAQ](FAQ.md). If you couldn't find help there, try searching our GitHub issues. We intend the issues page to be a forum in which the community collectively troubleshoots problems.

If bugs are found, **we appreciate pull requests** (including adding Q&A's to `FAQ.md` and improving our installation instructions and troubleshooting documents). Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information about contributing to Detectron.

## References

- [Data Distillation: Towards Omni-Supervised Learning](https://arxiv.org/abs/1712.04440).
  Ilija Radosavovic, Piotr Dollár, Ross Girshick, Georgia Gkioxari, and Kaiming He.
  Tech report, arXiv, Dec. 2017.
- [Learning to Segment Every Thing](https://arxiv.org/abs/1711.10370).
  Ronghang Hu, Piotr Dollár, Kaiming He, Trevor Darrell, and Ross Girshick.
  Tech report, arXiv, Nov. 2017.
- [Non-Local Neural Networks](https://arxiv.org/abs/1711.07971).
  Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
  Tech report, arXiv, Nov. 2017.
- [Mask R-CNN](https://arxiv.org/abs/1703.06870).
  Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick.
  IEEE International Conference on Computer Vision (ICCV), 2017.
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
  Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár.
  IEEE International Conference on Computer Vision (ICCV), 2017.
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677).
  Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He.
  Tech report, arXiv, June 2017.
- [Detecting and Recognizing Human-Object Interactions](https://arxiv.org/abs/1704.07333).
  Georgia Gkioxari, Ross Girshick, Piotr Dollár, and Kaiming He.
  Tech report, arXiv, Apr. 2017.
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144).
  Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431).
  Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [R-FCN: Object Detection via Region-based Fully Convolutional Networks](http://arxiv.org/abs/1605.06409).
  Jifeng Dai, Yi Li, Kaiming He, and Jian Sun.
  Conference on Neural Information Processing Systems (NIPS), 2016.
- [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385).
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/abs/1506.01497)
  Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
  Conference on Neural Information Processing Systems (NIPS), 2015.
- [Fast R-CNN](http://arxiv.org/abs/1504.08083).
  Ross Girshick.
  IEEE International Conference on Computer Vision (ICCV), 2015.
