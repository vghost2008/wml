from argparse import ArgumentParser
import pickle
import numpy as np
from object_detection2.metrics.build import *
from object_detection2.metrics.toolkit import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('result', help='Config file')
    parser.add_argument('--metrics', type=str,default="PrecisionAndRecall",help='metrics')
    parser.add_argument('--num_classes', type=int,default=0,help='num of classes')
    parser.add_argument('--beg_score_thr', type=float,default=0.1,help='begin test thr')
    parser.add_argument('--end_score_thr', type=float,default=1.0, help='end test thr')
    parser.add_argument('--score_step', type=float,default=0.05, help='score step')
    parser.add_argument('--classes_wise', type=bool,default=False, help='score step')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.result,"rb") as f:
        data = pickle.load(f)
    num_classes = args.num_classes
    if num_classes == 0:
        for d in data:
            gtlabels = d['gtlabels']
            max_label = np.max(gtlabels)
            num_classes = max(num_classes,max_label+1)
            labels = d['labels']
            max_label = np.max(labels)
            num_classes = max(num_classes,max_label+1)
    if args.classes_wise:
        metrics_cfg = dict(model_type=METRICS_REGISTRY.get(args.metrics),num_classes=num_classes,classes_begin_value=0)
        metrics = ClassesWiseModelPerformace(**metrics_cfg)
    else:
        metrics_cfg = dict(type=args.metrics,num_classes=num_classes,classes_begin_value=0)
        metrics = build_metrics(metrics_cfg)

    score = args.beg_score_thr
    best_value = -1
    best_string = ""
    best_score = 0
    while score<=args.end_score_thr:
        if args.classes_wise:
            metrics = ClassesWiseModelPerformace(**metrics_cfg)
        else:
            metrics = build_metrics(metrics_cfg)
        for d in data:
            keep = d['probability']>=score
            d['probability'] = d['probability'][keep]
            d['labels'] = d['labels'][keep]
            d['boxes'] = d['boxes'][keep]
            metrics(**d)
        print(f"Score threshold: {score}")
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
