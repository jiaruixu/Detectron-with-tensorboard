# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Functions for evaluating results computed for a json dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import numpy as np
import os
import uuid
import pandas as pd
import matplotlib.pyplot as plt

from pycocotools.cocoeval import COCOeval

from detectron.core.config import cfg
from detectron.utils.io import save_object
import detectron.utils.boxes as box_utils

logger = logging.getLogger(__name__)


def evaluate_masks(
    json_dataset,
    all_boxes,
    all_segms,
    output_dir,
    use_salt=True,
    cleanup=False
):
    res_file = os.path.join(
        output_dir, 'segmentations_' + json_dataset.name + '_results'
    )
    if use_salt:
        res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    _write_coco_segms_results_file(
        json_dataset, all_boxes, all_segms, res_file)
    # Only do evaluation on non-test sets (annotations are undisclosed on test)
    if json_dataset.name.find('test') == -1:
        coco_eval = _do_segmentation_eval(json_dataset, res_file, output_dir)
    else:
        coco_eval = None
    # Optionally cleanup results json file
    if cleanup:
        os.remove(res_file)
    return coco_eval


def _write_coco_segms_results_file(
    json_dataset, all_boxes, all_segms, res_file
):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "segmentation": [...],
    #   "score": 0.236}, ...]
    results = []
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        if cls_ind >= len(all_boxes):
            break
        cat_id = json_dataset.category_to_id_map[cls]
        results.extend(_coco_segms_results_one_category(
            json_dataset, all_boxes[cls_ind], all_segms[cls_ind], cat_id))
    logger.info(
        'Writing segmentation results json to: {}'.format(
            os.path.abspath(res_file)))
    with open(res_file, 'w') as fid:
        json.dump(results, fid)


def _coco_segms_results_one_category(json_dataset, boxes, segms, cat_id):
    results = []
    image_ids = json_dataset.COCO.getImgIds()
    image_ids.sort()
    assert len(boxes) == len(image_ids)
    assert len(segms) == len(image_ids)
    for i, image_id in enumerate(image_ids):
        dets = boxes[i]
        rles = segms[i]

        if isinstance(dets, list) and len(dets) == 0:
            continue

        dets = dets.astype(np.float)
        scores = dets[:, -1]

        results.extend(
            [{'image_id': image_id,
              'category_id': cat_id,
              'segmentation': rles[k],
              'score': scores[k]}
              for k in range(dets.shape[0])])

    return results


def _do_segmentation_eval(json_dataset, res_file, output_dir):
    coco_dt = json_dataset.COCO.loadRes(str(res_file))
    coco_eval = COCOeval(json_dataset.COCO, coco_dt, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    _log_detection_eval_metrics(json_dataset, coco_eval)
    eval_file = os.path.join(output_dir, 'segmentation_results.pkl')
    save_object(coco_eval, eval_file)
    logger.info('Wrote json eval results to: {}'.format(eval_file))
    return coco_eval


def evaluate_boxes(
    json_dataset, all_boxes, output_dir, use_salt=True, cleanup=False, checkpoint_iter=None, tblogger=None
):
    res_file = os.path.join(
        output_dir, 'bbox_' + json_dataset.name + '_results'
    )
    if use_salt:
        res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    _write_coco_bbox_results_file(json_dataset, all_boxes, res_file)
    # Only do evaluation on non-test sets (annotations are undisclosed on test)
    if json_dataset.name.find('test') == -1:
        coco_eval = _do_detection_eval(json_dataset, res_file, output_dir, checkpoint_iter=checkpoint_iter, tblogger=tblogger)
    else:
        coco_eval = None
    # Optionally cleanup results json file
    if cleanup:
        os.remove(res_file)
    return coco_eval


def _write_coco_bbox_results_file(json_dataset, all_boxes, res_file):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "bbox": [258.15,41.29,348.26,243.78],
    #   "score": 0.236}, ...]
    results = []
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        if cls_ind >= len(all_boxes):
            break
        cat_id = json_dataset.category_to_id_map[cls]
        results.extend(_coco_bbox_results_one_category(
            json_dataset, all_boxes[cls_ind], cat_id))
    logger.info(
        'Writing bbox results json to: {}'.format(os.path.abspath(res_file)))
    with open(res_file, 'w') as fid:
        json.dump(results, fid)


def _coco_bbox_results_one_category(json_dataset, boxes, cat_id):
    results = []
    image_ids = json_dataset.COCO.getImgIds()
    image_ids.sort()
    assert len(boxes) == len(image_ids)
    for i, image_id in enumerate(image_ids):
        dets = boxes[i]
        if isinstance(dets, list) and len(dets) == 0:
            continue
        dets = dets.astype(np.float)
        scores = dets[:, -1]
        xywh_dets = box_utils.xyxy_to_xywh(dets[:, 0:4])
        xs = xywh_dets[:, 0]
        ys = xywh_dets[:, 1]
        ws = xywh_dets[:, 2]
        hs = xywh_dets[:, 3]
        results.extend(
            [{'image_id': image_id,
              'category_id': cat_id,
              'bbox': [xs[k], ys[k], ws[k], hs[k]],
              'score': scores[k]} for k in range(dets.shape[0])])
    return results


def _do_detection_eval(json_dataset, res_file, output_dir, checkpoint_iter=None, tblogger=None):
    coco_dt = json_dataset.COCO.loadRes(str(res_file))
    coco_eval = COCOeval(json_dataset.COCO, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    _log_detection_eval_metrics(json_dataset, coco_eval, output_dir=output_dir, checkpoint_iter=checkpoint_iter, tblogger=tblogger)
    eval_file = os.path.join(output_dir, 'detection_results.pkl')
    save_object(coco_eval, eval_file)
    logger.info('Wrote json eval results to: {}'.format(eval_file))
    return coco_eval


def _log_detection_eval_metrics(json_dataset, coco_eval, output_dir=None, checkpoint_iter=None, tblogger=None):
    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95
    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    recall = coco_eval.eval['recall'][ind_lo:(ind_hi + 1), :, 0, 2]
    recall_default = np.mean(recall[recall > -1])

    rec_thresh = coco_eval.params.recThrs.reshape(-1)
    plt.figure()
    writer = pd.ExcelWriter('{}/score_of_checkpoint{}(All_area).xlsx'.format(output_dir, checkpoint_iter), engine='xlsxwriter')
    for iou in range(ind_lo, ind_hi + 1):
        pred_val = coco_eval.eval['precision'][iou, :, 0, 0, 2].reshape(-1)
        plt.plot(rec_thresh, pred_val)
        score = coco_eval.eval['scores'][iou, :, 0, 0, 2].reshape(-1)
        df = pd.DataFrame({'Recall': rec_thresh,
                           'Precision': pred_val,
                           'Scores': score})
        df.to_excel(writer, sheet_name="IoU_%.2f" % (0.50 + iou * 0.05))

    writer.save()
    plt.legend(['Iou = 0.50', 'Iou = 0.55', 'Iou = 0.60', 'Iou = 0.65', 'Iou = 0.70',
                'Iou = 0.75', 'Iou = 0.80', 'Iou = 0.85', 'Iou = 0.90', 'Iou = 0.95'],
               loc='lower left')
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.grid(True)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Precision-recall curve of checkpoint {} (All area)'.format(checkpoint_iter))
    plt.savefig('{}/Precision-recall-curve-of-checkpoint{}(All area).png'.format(output_dir, checkpoint_iter))
    # plt.show()
    plt.close()

    plt.figure()
    writer = pd.ExcelWriter('{}/score_of_checkpoint{}(small_area).xlsx'.format(output_dir, checkpoint_iter),
                            engine='xlsxwriter')
    for iou in range(ind_lo, ind_hi + 1):
        pred_val = coco_eval.eval['precision'][iou, :, 0, 1, 2].reshape(-1)
        plt.plot(rec_thresh, pred_val)
        score = coco_eval.eval['scores'][iou, :, 0, 1, 2].reshape(-1)
        df = pd.DataFrame({'Recall': rec_thresh,
                           'Precision': pred_val,
                           'Scores': score})
        df.to_excel(writer, sheet_name="IoU_%.2f" % (0.50 + iou * 0.05))

    writer.save()
    plt.legend(['Iou = 0.50', 'Iou = 0.55', 'Iou = 0.60', 'Iou = 0.65', 'Iou = 0.70',
                'Iou = 0.75', 'Iou = 0.80', 'Iou = 0.85', 'Iou = 0.90', 'Iou = 0.95'],
               loc='lower left')
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.grid(True)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Precision-recall curve of checkpoint {} (small area)'.format(checkpoint_iter))
    plt.savefig('{}/Precision-recall-curve-of-checkpoint{}(small area).png'.format(output_dir, checkpoint_iter))
    plt.close()

    plt.figure()
    writer = pd.ExcelWriter('{}/score_of_checkpoint{}(medium_area).xlsx'.format(output_dir, checkpoint_iter),
                            engine='xlsxwriter')
    for iou in range(ind_lo, ind_hi + 1):
        pred_val = coco_eval.eval['precision'][iou, :, 0, 2, 2].reshape(-1)
        plt.plot(rec_thresh, pred_val)
        score = coco_eval.eval['scores'][iou, :, 0, 2, 2].reshape(-1)
        df = pd.DataFrame({'Recall': rec_thresh,
                           'Precision': pred_val,
                           'Scores': score})
        df.to_excel(writer, sheet_name="IoU_%.2f" % (0.50 + iou * 0.05))

    writer.save()
    plt.legend(['Iou = 0.50', 'Iou = 0.55', 'Iou = 0.60', 'Iou = 0.65', 'Iou = 0.70',
                'Iou = 0.75', 'Iou = 0.80', 'Iou = 0.85', 'Iou = 0.90', 'Iou = 0.95'],
               loc='lower left')
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.grid(True)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Precision-recall curve of checkpoint {} (medium area)'.format(checkpoint_iter))
    plt.savefig('{}/Precision-recall-curve-of-checkpoint{}(medium area).png'.format(output_dir, checkpoint_iter))
    plt.close()

    plt.figure()
    writer = pd.ExcelWriter('{}/score_of_checkpoint{}(large_area).xlsx'.format(output_dir, checkpoint_iter),
                            engine='xlsxwriter')
    for iou in range(ind_lo, ind_hi + 1):
        pred_val = coco_eval.eval['precision'][iou, :, 0, 3, 2].reshape(-1)
        plt.plot(rec_thresh, pred_val)
        score = coco_eval.eval['scores'][iou, :, 0, 3, 2].reshape(-1)
        df = pd.DataFrame({'Recall': rec_thresh,
                           'Precision': pred_val,
                           'Scores': score})
        df.to_excel(writer, sheet_name="IoU_%.2f" % (0.50 + iou * 0.05))

    writer.save()
    plt.legend(['Iou = 0.50', 'Iou = 0.55', 'Iou = 0.60', 'Iou = 0.65', 'Iou = 0.70',
                'Iou = 0.75', 'Iou = 0.80', 'Iou = 0.85', 'Iou = 0.90', 'Iou = 0.95'],
               loc='lower left')
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.grid(True)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Precision-recall curve of checkpoint {} (large area)'.format(checkpoint_iter))
    plt.savefig('{}/Precision-recall-curve-of-checkpoint{}(large area).png'.format(output_dir, checkpoint_iter))
    plt.close()


    # write iou of each annotation to excel
    writer = pd.ExcelWriter('{}/ious_checkpoint{}.xlsx'.format(output_dir, checkpoint_iter),
                            engine='xlsxwriter')
    ious = coco_eval.ious
    # dts = coco_eval._dts
    # gts = coco_eval._gts
    image_id_list = []
    image_name_list = []
    x_list = []
    y_list = []
    w_list = []
    h_list = []
    area_list = []
    score_list = []
    iou_list = []
    gt_x_list = []
    gt_y_list = []
    gt_w_list = []
    gt_h_list = []
    gt_area_list = []

    for image_id in coco_eval.params.imgIds:
        cat_id = 1
        dts = coco_eval._dts[image_id, cat_id]
        gts = coco_eval._gts[image_id, cat_id]
        dtind = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in dtind]
        gtind = np.argsort([g['_ignore'] for g in gts], kind='mergesort')
        gts = [gts[i] for i in gtind]
        if len(gts) == 0 or len(dts) == 0:
            continue
        if len(dts) > 100:
            dts = dts[0:100]

        iou_image_anns = ious[image_id, 1]
        scorem = np.zeros(len(gts))
        for ann_ind in range(len(iou_image_anns)):
            iou_dt_gt_array = iou_image_anns[ann_ind]
            max_iou_gt_ind = np.argmax(iou_dt_gt_array)
            max_iou = max(iou_dt_gt_array)
            dt = dts[ann_ind]
            gt = gts[max_iou_gt_ind]

            if max_iou != 0 and dt['score'] > scorem[max_iou_gt_ind]:
                scorem[max_iou_gt_ind] = dt['score']
                image_id_list.append(image_id)
                image_name_list.append(coco_eval.cocoDt.imgs[image_id]['file_name'])
                x_list.append(dt['bbox'][0])
                y_list.append(dt['bbox'][1])
                w_list.append(dt['bbox'][2])
                h_list.append(dt['bbox'][3])
                area_list.append(dt['area'])
                iou_list.append(max_iou)
                score_list.append(dt['score'])
                gt_x_list.append(gt['bbox'][0])
                gt_y_list.append(gt['bbox'][1])
                gt_w_list.append(gt['bbox'][2])
                gt_h_list.append(gt['bbox'][3])
                gt_area_list.append(gt['area'])


    df = pd.DataFrame({'Image id': image_id_list,
                       'Image name': image_name_list,
                       'bbox_x': x_list,
                       'bbox_y': y_list,
                       'bbox_w': w_list,
                       'bbox_h': h_list,
                       'iou': iou_list,
                       'area': area_list,
                       'score': score_list,
                       'gt_bbox_x': gt_x_list,
                       'gt_bbox_y': gt_y_list,
                       'gt_bbox_w': gt_w_list,
                       'gt_bbox_h': gt_h_list,
                       'gt_area': gt_area_list})
    columns = ['Image id','Image name','bbox_x','bbox_y', 'bbox_w', 'bbox_h',
                'area','iou','score','gt_bbox_x','gt_bbox_y','gt_bbox_w','gt_bbox_h','gt_area']
    df.to_excel(writer, columns=columns)
    writer.save()


    if tblogger:
        tblogger.write_scalars({"average_precision": ap_default}, checkpoint_iter)
        tblogger.write_scalars({"average_recall": recall_default}, checkpoint_iter)
        # tb_log_stats(tblogger, ap_default, checkpoint_iter)
        # tb_log_stats(tblogger, recall_default, checkpoint_iter)

    logger.info(
        '~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] ~~~~'.format(
            IoU_lo_thresh, IoU_hi_thresh))
    logger.info('{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        # minus 1 because of __background__
        precision = coco_eval.eval['precision'][
            ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
        ap = np.mean(precision[precision > -1])
        logger.info('{:.1f}'.format(100 * ap))
    logger.info('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()

def tb_log_stats(tblogger, stats, cur_iter):
    """Log the tracked statistics to tensorboard"""
    for k in stats:
        v = stats[k]
        if isinstance(v, dict):
            tb_log_stats(tblogger, v, cur_iter)
        else:
            tblogger.write_scalars({k: v}, cur_iter)

def evaluate_box_proposals(
    json_dataset, roidb, thresholds=None, area='all', limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        'all': 0,
        'small': 1,
        'medium': 2,
        'large': 3,
        '96-128': 4,
        '128-256': 5,
        '256-512': 6,
        '512-inf': 7}
    area_ranges = [
        [0**2, 1e5**2],    # all
        [0**2, 32**2],     # small
        [32**2, 96**2],    # medium
        [96**2, 1e5**2],   # large
        [96**2, 128**2],   # 96-128
        [128**2, 256**2],  # 128-256
        [256**2, 512**2],  # 256-512
        [512**2, 1e5**2]]  # 512-inf
    assert area in areas, 'Unknown area range: {}'.format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = np.zeros(0)
    num_pos = 0
    for entry in roidb:
        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
        gt_boxes = entry['boxes'][gt_inds, :]
        gt_areas = entry['seg_areas'][gt_inds]
        valid_gt_inds = np.where(
            (gt_areas >= area_range[0]) & (gt_areas <= area_range[1]))[0]
        gt_boxes = gt_boxes[valid_gt_inds, :]
        num_pos += len(valid_gt_inds)
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        boxes = entry['boxes'][non_gt_inds, :]
        if boxes.shape[0] == 0:
            continue
        if limit is not None and boxes.shape[0] > limit:
            boxes = boxes[:limit, :]
        overlaps = box_utils.bbox_overlaps(
            boxes.astype(dtype=np.float32, copy=False),
            gt_boxes.astype(dtype=np.float32, copy=False))
        _gt_overlaps = np.zeros((gt_boxes.shape[0]))
        for j in range(min(boxes.shape[0], gt_boxes.shape[0])):
            # find which proposal box maximally covers each gt box
            argmax_overlaps = overlaps.argmax(axis=0)
            # and get the iou amount of coverage for each gt box
            max_overlaps = overlaps.max(axis=0)
            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ind = max_overlaps.argmax()
            gt_ovr = max_overlaps.max()
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1
        # append recorded iou coverage level
        gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

    gt_overlaps = np.sort(gt_overlaps)
    if thresholds is None:
        step = 0.05
        thresholds = np.arange(0.5, 0.95 + 1e-5, step)
    recalls = np.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
            'gt_overlaps': gt_overlaps, 'num_pos': num_pos}


def evaluate_keypoints(
    json_dataset,
    all_boxes,
    all_keypoints,
    output_dir,
    use_salt=True,
    cleanup=False
):
    res_file = os.path.join(
        output_dir, 'keypoints_' + json_dataset.name + '_results'
    )
    if use_salt:
        res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    _write_coco_keypoint_results_file(
        json_dataset, all_boxes, all_keypoints, res_file)
    # Only do evaluation on non-test sets (annotations are undisclosed on test)
    if json_dataset.name.find('test') == -1:
        coco_eval = _do_keypoint_eval(json_dataset, res_file, output_dir)
    else:
        coco_eval = None
    # Optionally cleanup results json file
    if cleanup:
        os.remove(res_file)
    return coco_eval


def _write_coco_keypoint_results_file(
    json_dataset, all_boxes, all_keypoints, res_file
):
    results = []
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        if cls_ind >= len(all_keypoints):
            break
        logger.info(
            'Collecting {} results ({:d}/{:d})'.format(
                cls, cls_ind, len(all_keypoints) - 1))
        cat_id = json_dataset.category_to_id_map[cls]
        results.extend(_coco_kp_results_one_category(
            json_dataset, all_boxes[cls_ind], all_keypoints[cls_ind], cat_id))
    logger.info(
        'Writing keypoint results json to: {}'.format(
            os.path.abspath(res_file)))
    with open(res_file, 'w') as fid:
        json.dump(results, fid)


def _coco_kp_results_one_category(json_dataset, boxes, kps, cat_id):
    results = []
    image_ids = json_dataset.COCO.getImgIds()
    image_ids.sort()
    assert len(kps) == len(image_ids)
    assert len(boxes) == len(image_ids)
    use_box_score = False
    if cfg.KRCNN.KEYPOINT_CONFIDENCE == 'logit':
        # This is ugly; see utils.keypoints.heatmap_to_keypoints for the magic
        # indexes
        score_index = 2
    elif cfg.KRCNN.KEYPOINT_CONFIDENCE == 'prob':
        score_index = 3
    elif cfg.KRCNN.KEYPOINT_CONFIDENCE == 'bbox':
        use_box_score = True
    else:
        raise ValueError(
            'KRCNN.KEYPOINT_CONFIDENCE must be "logit", "prob", or "bbox"')
    for i, image_id in enumerate(image_ids):
        if len(boxes[i]) == 0:
            continue
        kps_dets = kps[i]
        scores = boxes[i][:, -1].astype(np.float)
        if len(kps_dets) == 0:
            continue
        for j in range(len(kps_dets)):
            xy = []

            kps_score = 0
            for k in range(kps_dets[j].shape[1]):
                xy.append(float(kps_dets[j][0, k]))
                xy.append(float(kps_dets[j][1, k]))
                xy.append(1)
                if not use_box_score:
                    kps_score += kps_dets[j][score_index, k]

            if use_box_score:
                kps_score = scores[j]
            else:
                kps_score /= kps_dets[j].shape[1]

            results.extend([{'image_id': image_id,
                             'category_id': cat_id,
                             'keypoints': xy,
                             'score': kps_score}])
    return results


def _do_keypoint_eval(json_dataset, res_file, output_dir):
    ann_type = 'keypoints'
    imgIds = json_dataset.COCO.getImgIds()
    imgIds.sort()
    coco_dt = json_dataset.COCO.loadRes(res_file)
    coco_eval = COCOeval(json_dataset.COCO, coco_dt, ann_type)
    coco_eval.params.imgIds = imgIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    eval_file = os.path.join(output_dir, 'keypoint_results.pkl')
    save_object(coco_eval, eval_file)
    logger.info('Wrote json eval results to: {}'.format(eval_file))
    coco_eval.summarize()
    return coco_eval
