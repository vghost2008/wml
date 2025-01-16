from wml.thirdparty.registry import Registry
import copy

METRICS_REGISTRY = Registry("Metrics")
CLASSIFIER_METRICS_REGISTRY = Registry("ClassifierMetrics")

def build_metrics(cfg):
    cfg = copy.deepcopy(cfg)
    type_str = cfg.pop("type")
    return METRICS_REGISTRY.get(type_str)(**cfg)

def build_classifier_metrics(cfg):
    cfg = copy.deepcopy(cfg)
    type_str = cfg.pop("type")
    return CLASSIFIER_METRICS_REGISTRY.get(type_str)(**cfg)
