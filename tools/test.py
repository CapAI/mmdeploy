import argparse
import sys

from mmcv import DictAction
from mmcv.parallel import MMDataParallel

from mmdeploy.apis import (build_dataloader, build_dataset, init_backend_model,
                           post_process_outputs, single_gpu_test)
from mmdeploy.utils.config_utils import get_codebase, load_config
from mmdeploy.utils.timer import TimeCounter


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDeploy test (and eval) a backend.')
    parser.add_argument('deploy_cfg', help='Deploy config path')
    parser.add_argument('model_cfg', help='Model config path')
    parser.add_argument(
        '--model', type=str, nargs='+', help='Input model files.')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the codebase and the '
        'dataset, e.g., "bbox", "segm", "proposal" for COCO, and "mAP", '
        '"recall" for PASCAL VOC in mmdet; "accuracy", "precision", "recall", '
        '"f1_score", "support" for single label dataset, and "mAP", "CP", "CR"'
        ', "CF1", "OP", "OR", "OF1" for multi-label dataset in mmcls')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--speed-test', action='store_true', help='activate speed test')
    parser.add_argument(
        '--warmup',
        type=int,
        help='warmup before counting inference elaps, require setting '
        'speed-test first',
        default=10)
    parser.add_argument(
        '--log-interval',
        type=int,
        help='the interval between each log, require setting '
        'speed-test first',
        default=100)
    parser.add_argument(
        '--log2file',
        type=str,
        help='log speed in file format ,need speed-test first')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg

    # load deploy_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    # prepare the dataset loader
    codebase = get_codebase(deploy_cfg)
    dataset_type = 'test'
    dataset = build_dataset(codebase, model_cfg, dataset_type)
    data_loader = build_dataloader(
        codebase,
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=model_cfg.data.workers_per_gpu)

    # load the model of the backend
    device_id = -1 if args.device == 'cpu' else 0
    model = init_backend_model(
        args.model,
        model_cfg=args.model_cfg,
        deploy_cfg=args.deploy_cfg,
        device_id=device_id)

    model = MMDataParallel(model, device_ids=[0])
    if args.speed_test:
        with_sync = device_id == 0
        output_file = sys.stdout
        if args.log2file:
            output_file = args.log2file

        with TimeCounter.activate(
                warmup=args.warmup,
                log_interval=args.log_interval,
                with_sync=with_sync,
                file=output_file):
            outputs = single_gpu_test(codebase, model, data_loader, args.show,
                                      args.show_dir, args.show_score_thr)
    else:
        outputs = single_gpu_test(codebase, model, data_loader, args.show,
                                  args.show_dir, args.show_score_thr)
    post_process_outputs(outputs, dataset, model_cfg, codebase, args.metrics,
                         args.out, args.metric_options, args.format_only)


if __name__ == '__main__':
    main()