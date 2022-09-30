#coding=utf-8
from wcollections import *
from wfilesystem import *
import numpy as np
import random
import time
from functools import wraps
import sys
import datetime
import hashlib


def _to_chinese_num(i,numbers,unites):
    j = i%10
    i = i/10
    res = numbers[j]
    if unites[0] is not None and len(unites)>0:
        res = res+unites[0]
    if i>0:
        return _to_chinese_num(i,numbers,unites[1:])+res
    else:
        return res

def to_chinese_num(i):
    if i==0:
        return "零"
    unites=[None,"十","百","千","万","十万","千万","亿","十亿"]
    numbers=["","一","二","三","四","五","六","七","八","九"]
    res = _to_chinese_num(i,numbers,unites)
    if res.startswith("一十"):
        res = res.decode("utf-8")
        res = res[1:]
        res = res.encode("utf-8")
    return res


def show_member(obj,name=None):
    if name is not None:
        print("Show %s."%(name))

    for name,var in vars(obj).items():
        print("%s : "%(name),var)

def show_list(values,fmt=None,recurse=False):
    if values is None:
        return
    if isinstance(values,str):
        return show_list([values])
    print("[")
    if fmt is None:
        for v in values:
            if recurse and isinstance(v,(list,tuple)):
                show_list(v)
            else:
                print(v)
    else:
        for v in values:
            if recurse and isinstance(v,(list,tuple)):
                show_list(v)
            else:
                print(fmt.format(v))

    print("]")

def show_dict(values):
    print("{")
    for k,v in values.items():
        print(k,":",v,",")
    print("}")

def nparray2str(value,split=",",format="{}"):
    if not isinstance(value,np.ndarray):
        value = np.array(value)
    ndims = len(value.shape)
    if ndims == 1:
        r_str = "["
        for x in value[:-1]:
            r_str+=format.format(x)+split
        r_str+=format.format(value[-1])+"]"
        return r_str
    else:
        r_str = "["
        for x in value[:-1]:
            r_str+=nparray2str(x,split=split,format=format)+split+"\n"
        r_str+=nparray2str(value[-1],split=split,format=format)+"]\n"
        return r_str

def show_nparray(value,name=None,split=",",format="{}"):
    if name is not None:
        print(name)
    print(nparray2str(value,split=split,format=format))


def reduce_prod(x):
    if len(x)==0:
        return 0
    elif len(x)==1:
        return x[0]
    res = x[0]
    for v in x[1:]:
        res *= v

    return res

def any(iterable,v=None):
    if v is None:
        for value in iterable:
            if value is None:
                return True
        return False
    else:
        t = type(v)
        for value in iterable:
            if isinstance(t,value) and v==value:
                return True

        return False

def all(iterable,v=None):
    if v is None:
        for value in iterable:
            if value is not None:
                return False
        return True
    else:
        t = type(v)
        for value in iterable:
            if (not isinstance(t,value)) or v!=value:
                return False

        return True

def gather(data,indexs):
    res_data = []
    
    for d in indexs:
        res_data.append(data[d])
    
    return res_data

class TimeThis():
    def __init__(self,name="TimeThis",auto_show=True):
        self.begin_time = time.time()
        self.end_time = 0
        self.name = name
        self.auto_show = auto_show
        self.idx = 0

    def __enter__(self):
        self.begin_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.auto_show:
            te = (self.end_time-self.begin_time)*1000
            fps = 1000/(te+1e-8)
            print(f"{self.name}: total time {te:.3f}, FPS={fps:.3f}.")

    def time(self):
        return self.end_time-self.begin_time

    def log_and_restart(self,sub_name=""):
        self.end_time = time.time()
        te = (self.end_time - self.begin_time) * 1000
        fps = 1000 / (te + 1e-8)
        print(f"{self.name}:{self.idx}:{sub_name}: total time {te:.3f}, FPS={fps:.3f}.")
        self.begin_time = self.end_time
        self.idx += 1


class AvgTimeThis():
    def __init__(self,name="TimeThis",skip_nr=3):
        self.step = 0
        self.begin_time = 0.
        self.name = name
        self.datas = []
        self.skip_nr = skip_nr

    def __enter__(self):
        self.begin_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.step = self.step + 1
        if(self.step>self.skip_nr):
            self.datas.append(time.time()-self.begin_time)

    def clear(self):
        self.datas = []

    def get_time(self):
        if(len(self.datas) == 0):
                return 0
        return np.mean(np.array(self.datas))

    def __str__(self):
        return "AVG: "+str(self.get_time())+ f", test_nr = {self.step}, {self.datas}"

class MovingAvg(object):
    def __init__(self,init_val = 0.0,momentum=0.99):
        self.v = init_val
        self.momentum = momentum

    def __call__(self, v):
        self.v = self.v*self.momentum+v*(1-self.momentum)

    def value(self):
        return self.v


class EstimateTimeCost(object):
    def __init__(self,total_nr=1,auto_log=False):
        self.begin_time = None
        self.total_nr = total_nr
        self.process_nr = 0
        self.reset()
        self.auto_log = auto_log

    def reset(self,total_nr = None):
        self.begin_time = time.time()
        if total_nr is not None:
            self.total_nr = total_nr
        self.process_nr = 0

    def add_count(self):
        self.process_nr += 1
        return self.__repr__()
    
    def set_process_nr(self,process_nr):
        self.process_nr = process_nr
        return self.__repr__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process_nr += 1
        if self.auto_log:
            print(self.__str__())

    def __repr__(self):
        left_time = ((time.time() - self.begin_time) / max(self.process_nr, 1)) * (
                    self.total_nr- self.process_nr)
        d = datetime.datetime.now() + datetime.timedelta(seconds=left_time)
        res = f"already use {(time.time() - self.begin_time) / 3600:.3f}h, {left_time / 3600:.3f}h left, expected to be finished at {str(d)}"
        return res

def time_this(func):
    @wraps(func)
    def wraps_func(*args,**kwargs):
        begin_t = time.time()
        res = func(*args,**kwargs)
        print(f"Time cost {time.time()-begin_t}.")
        return res
    return wraps_func

def no_throw(func):
    @wraps(func)
    def wraps_func(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except:
            return None
    return wraps_func
'''
将list data分为多个组，每个组size个元素
'''
def list_to_2dlist(data,size):
    data_nr = len(data)
    if data_nr<size:
        return [data]
    res = []
    index = 0
    while index<data_nr:
       end_index = min(index+size,data_nr)
       res.append(data[index:end_index])
       index = end_index
    return res

'''
将list data分nr个组，每个组的元素数量基本相等
'''
def list_to_2dlistv2(data,nr):
    size = (len(data)+nr-1)//nr
    return list_to_2dlist(data,size)

def random_uniform(minmaxs):
    res = []
    for min,max in minmaxs:
        res.append(random.uniform(min,max))
    return res

def random_uniform_indict(minmaxs):
    res = {}
    for key,data in minmaxs.items():
        if len(data) == 2:
            if isinstance(data[0],bool):
                index = random.randint(0,1)
                res[key] = data[index]
            else:
                min,max = data[0],data[1]
                res[key] = random.uniform(min,max)
        else:
            index = random.randint(0,len(data)-1)
            res[key] = data[index]

    return res

def is_child_of(obj, cls):
    try:
        for i in obj.__bases__:
            if i is cls or isinstance(i, cls):
                return True
        for i in obj.__bases__:
            if is_child_of(i, cls):
                return True
    except AttributeError:
        return is_child_of(obj.__class__, cls)
    return False


def add_dict(lhv,rhv):
    res = dict(lhv)
    for k,v in rhv.items():
        if k in res:
            res[k] = res[k]+v
        else:
            res[k] = v
    return res


def sleep_until(runtime):
    target_datetime = datetime.datetime.strptime(runtime, "%Y-%m-%d %H:%M:%S")
    while True:
        wait_time = (target_datetime - datetime.datetime.now()).total_seconds() / 3600.0
        sys.stdout.write(
            f"\rRumtime is {target_datetime}, current datetime is {datetime.datetime.now()}, need to wait for {wait_time:.2f}h", )
        sys.stdout.flush()

        if datetime.datetime.now() >= target_datetime:
            break
        else:
            time.sleep(30)

def file_md5(path):
    with open(path,'rb') as f:
        data = f.read()
    return hashlib.md5(data).hexdigest()

def nparray(data,default_shape=[0],dtype=np.float):
    res = np.array(data)
    if res.size == 0:
        return np.zeros(default_shape,dtype=dtype)
    return res

