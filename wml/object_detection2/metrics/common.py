from abc import ABCMeta, abstractmethod

class BaseMetrics(metaclass=ABCMeta):
    def __init__(self,cfg_name="",**kwargs) -> None:
        self._current_info = ""
        self.cfg_name = cfg_name
        pass

    def current_info(self):
        return self._current_info

    def __repr__(self):
        return self.to_string()

    @abstractmethod
    def show(self):
        pass

    def value(self):
        pass

    def detail_valus(self):
        return self.value()
    

def safe_persent(v0,v1):
    if v1==0:
        return 100.
    else:
        return v0*100./v1

def safe_score(v0,v1,max_v = 1.0):
    if v1==0:
        return max_v
    else:
        return v0*max_v/v1

class ComposeMetrics(BaseMetrics):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.metrics = list(args)+list(kwargs.values())

    def __call__(self, *args,**kwargs):
        [m(*args,**kwargs) for m in self.metrics]
        #self._current_info = "; ".join([m.current_info() for m in self.metrics])

    def evaluate(self):
        [m.evaluate() for m in self.metrics]

    def show(self,name=""):
        [m.show(name=name) for m in self.metrics]

    def to_string(self):
        return ";".join([m.to_string() for m in self.metrics])

    def value(self):
        return self.metrics[0].value()

class BaseClassifierMetrics(metaclass=ABCMeta):
    def __init__(self,*args,**kwargs):
        self._current_info = ""
        pass

    def value(self):
        pass

    def to_string(self):
        return str(self.value())

    def __repr__(self):
        return self.to_string()

    def mark_down(self,name=""):
        print(name,self.to_string())

    def current_info(self):
        return self._current_info