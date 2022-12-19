from thirdparty.registry import Registry
import copy

METRICS_REGISTRY = Registry("Metrics")

def build_metrics(cfg):
    cfg = copy.deepcopy(cfg)
    type_str = cfg.pop("type")
    return METRICS_REGISTRY.get(type_str)(**cfg)

