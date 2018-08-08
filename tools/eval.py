"""Evaluate a network with Detectron."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import pprint
import sys
import time
import re

from copy import deepcopy
from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.core.test_engine import run_inference
from detectron.utils.logging import setup_logging
import detectron.utils.c2 as c2_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# gloabl value
file_processed = []

def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wait',
        dest='wait',
        help='wait until net file exists',
        default=True,
        type=bool
    )
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true'
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='using cfg.NUM_GPUS for inference',
        action='store_true'
    )
    parser.add_argument(
        '--range',
        dest='range',
        help='start (inclusive) and end (exclusive) indices',
        default=None,
        type=int,
        nargs=2
    )
    parser.add_argument(
        '--use_tfboard',
        dest='use_tfboard',
        help='Use tensorboard to log training info',
        action='store_false'
    )
    parser.add_argument(
        'opts',
        help='See detectron/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def checkNewCheckpoint(args, cfg, logger):
    global file_processed
    while 1:
        output_dir = cfg.TEST.WEIGHTS
        files = os.listdir(output_dir)
        file_not_processed = list(set(files).difference(set(file_processed)))
        if len(file_not_processed) != 0:
            logger.info('{} evaluating...'.format(time.ctime()))
            eval(args, cfg, logger, file_not_processed)
            file_processed = deepcopy(files)

        logger.info('{} waiting for new checkpoint...'.format(time.ctime()))
        time.sleep(1520)

def eval(args, cfg, logger, files):
    for f in files:
        iter_string = re.findall(r'(?<=model_iter)\d+(?=\.pkl)', f)
        if len(iter_string) > 0:  ## and (int(iter_string[0]) == 34999):
            checkpoint_iter = int(iter_string[0])
            resume_weights_file = f
            weights_file = os.path.join(cfg.TEST.WEIGHTS, resume_weights_file)
            logger.info(
                '========> Resuming from checkpoint {} at iter {}'.
                    format(weights_file, checkpoint_iter)
            )

            run_inference(
                weights_file,
                ind_range=args.range,
                multi_gpu_testing=args.multi_gpu_testing,
                check_expected_results=True,
                checkpoint_iter=checkpoint_iter,
                use_tfboard=args.use_tfboard if args.use_tfboard else None,
            )


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logger = setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    assert_and_infer_cfg()
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    checkNewCheckpoint(args, cfg, logger)
