import sys
import os
import argparse
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import thirdparty.TrackEval.trackeval as trackeval  # noqa: E402

def eval_two_dirs(gt_floder,trackers_floder,metrics=["HOTA"]):
    config_args = {}
    config_args["GT_IMG_LOC_FORMAT"] = ""
    config_args["GT_LOC_FORMAT"] = "{gt_folder}/{seq}.txt"

    return eval(gt_floder=gt_floder,trackers_floder=trackers_floder,metrics=metrics,config_args=config_args)

def eval(gt_floder,trackers_floder,metrics=["HOTA"],config_args={}):
    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    args = {}
    args["BENCHMARK"] = ""
    args["SPLIT_TO_EVAL"] = ""
    args["TRACKERS_TO_EVAL"] = [""]
    args["METRICS"] = metrics
    args["USE_PARALLEL"] = "False"
    args["NUM_PARALLEL_CORES"] = 1
    args["GT_FOLDER"] = gt_floder
    args["TRACKERS_FOLDER"] = trackers_floder
    
    config["SKIP_SPLIT_FOL"] = True
    config["TRACKER_SUB_FOLDER"] = ""
    config.update(config_args)

    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            elif setting == 'SEQ_INFO':
                x = dict(zip(args[setting], [None]*len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    data = evaluator.evaluate(dataset_list, metrics_list)
    return data

def eval_old(gt_floder,trackers_floder,BENCHMARK="sportsmot",split="val",trackers_to_eval="MPNTrack",metrics=["HOTA"]):
    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    args = {}
    args["BENCHMARK"] = BENCHMARK
    args["SPLIT_TO_EVAL"] = split
    args["TRACKERS_TO_EVAL"] = [trackers_to_eval]
    args["METRICS"] = metrics
    args["USE_PARALLEL"] = "False"
    args["NUM_PARALLEL_CORES"] = 1
    args["GT_FOLDER"] = gt_floder
    args["TRACKERS_FOLDER"] = trackers_floder

    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            elif setting == 'SEQ_INFO':
                x = dict(zip(args[setting], [None]*len(args[setting])))
            else:
                x = args[setting]
            config[setting] = x
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    data = evaluator.evaluate(dataset_list, metrics_list)
    return data
