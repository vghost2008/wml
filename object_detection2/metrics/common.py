from abc import ABCMeta, abstractclassmethod

class BaseMetrics(metaclass=ABCMeta):
    def __init__(self,cfg_name="",**kwargs) -> None:
        self._current_info = ""
        self.cfg_name = cfg_name
        pass

    def current_info(self):
        return self._current_info

    def __repr__(self):
        return self.to_string()

    @abstractclassmethod
    def show(self):
        pass
    

def safe_persent(v0,v1):
    if v1==0:
        return 100.
    else:
        return v0*100./v1

class ComposeMetrics(BaseMetrics):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.metrics = list(args)+list(kwargs.values())

    def __call__(self, *args,**kwargs):
        [m(*args,**kwargs) for m in self.metrics]
        self._current_info = "; ".join([m.current_info() for m in self.metrics])

    def evaluate(self):
        [m.evaluate() for m in self.metrics]

    def show(self,name=""):
        [m.show(name=name) for m in self.metrics]

    def to_string(self):
        return ";".join([m.to_string() for m in self.metrics])

    def value(self):
        return self.metrics[0].value()