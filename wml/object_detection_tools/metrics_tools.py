from argparse import ArgumentParser
import pickle
import numpy as np
from wml.object_detection2.metrics.build import *
from wml.object_detection2.metrics.toolkit import *
from wml.object_detection2.standard_names import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('result', help='Config file')
    parser.add_argument('--metrics', type=str,default="PrecisionAndRecall",help='metrics')
    parser.add_argument('--num-classes', type=int,default=0,help='num of classes')
    parser.add_argument('--beg-score-thr', type=float,default=0.1,help='begin test thr')
    parser.add_argument('--end-score-thr', type=float,default=1.0, help='end test thr')
    parser.add_argument('--score-step', type=float,default=0.05, help='score step')
    parser.add_argument('--classes-wise', action='store_true', help='score step')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    parser.add_argument('--iou-thr', type=float, help='iou thr')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.result,"rb") as f:
        data = pickle.load(f)
    num_classes = args.num_classes
    if num_classes == 0:
        for d in data:
            try:
                gtlabels = d['gtlabels']
                max_label = np.max(gtlabels)
                num_classes = max(num_classes,max_label+1)
                labels = d['labels']
                max_label = np.max(labels)
                num_classes = max(num_classes,max_label+1)
            except:
                pass
        print(f"Auto update num_classes to {num_classes}")
    if args.classes_wise:
        metrics_cfg = dict(model_type=METRICS_REGISTRY.get(args.metrics),num_classes=num_classes,classes_begin_value=0)
        if args.iou_thr is not None and args.iou_thr >0:
            metrics_cfg['model_args'] = dict(threshold = args.iou_thr)
        metrics = ClassesWiseModelPerformace(**metrics_cfg)
    else:
        metrics_cfg = dict(type=args.metrics,num_classes=num_classes,classes_begin_value=0)
        if args.iou_thr is not None and args.iou_thr >0:
            metrics_cfg['threshold'] = args.iou_thr
        metrics = build_metrics(metrics_cfg)

    score = args.beg_score_thr
    best_value = -1
    best_string = ""
    best_score = 0
    print(f"Total {len(data)} samples.")
    while score<=args.end_score_thr:
        if args.classes_wise:
            metrics = ClassesWiseModelPerformace(**metrics_cfg)
        else:
            metrics = build_metrics(metrics_cfg)
        for d in data:
            file = d.pop('file',"")
            if 'probability' in d:
                scores_key = 'probability'
            elif 'scores' in d:
                scores_key = 'scores'
            else:
                raise RuntimeError(f"Find scores key faind: {list(d.keys())}")
            keep = d[scores_key]>=score
            d[scores_key] = d[scores_key][keep]
            d['labels'] = d['labels'][keep]
            if BOXES in d:
                d['boxes'] = d['boxes'][keep]
            if "kps" in d:
                d["kps"] = d["kps"][keep]
            metrics(**d)
            cur_info = metrics.current_info()
            if args.verbose and  len(cur_info)>0:
                print(file,cur_info)
        print(f"Score threshold: {score:.2f}")
        metrics.show()
        try:
            if hasattr(metrics,"value") and metrics.value()>best_value:
                best_value = metrics.value()
                best_string = str(metrics)
                best_score = score
        except:
            pass
        score += args.score_step
    
    if best_value>0:
        print(f"Best score {best_score}")
        print(best_string)
