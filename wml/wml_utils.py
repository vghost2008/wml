#coding=utf-8
from wml.wcollections import *
from wml.wfilesystem import *
from wml.walgorithm import *
import numpy as np
import random
import time
from functools import wraps
import sys
import datetime
import hashlib
import math
import pickle
import os
import subprocess
import warnings
from packaging.version import parse
from collections import OrderedDict
try:
    from importlib import metadata
except:
    metadata = None
    pass
import re
import torch


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
            if recurse and isinstance(v,(list,tuple,np.ndarray)):
                show_list(v)
            else:
                print(v)
    else:
        for v in values:
            if recurse and isinstance(v,(list,tuple,np.ndarray)):
                show_list(v)
            else:
                print(fmt.format(v))

    print("]")

def show_dict(values,format:str=None):
    print("{")
    for k,v in values.items():
        if format is None:
            print(k,":",v,",")
        else:
            print(k,":",format.format(v),",")
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
            print(f"{self.name}: total time {te:.3f}ms, FPS={fps:.3f}.")

    def time(self,reset=False):
        self.end_time = time.time()
        r = self.end_time-self.begin_time
        if reset:
            self.reset()
        return r

    def reset(self):
        self.begin_time = time.time()


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
    
    def max(self):
        return np.max(self.datas)

    def min(self):
        return np.min(self.datas)

    def __str__(self):
        return "AVG: "+str(self.get_time())+ f", test_nr = {self.step}"

class MTimer():
    def __init__(self,name="TimeThis",auto_show=True):
        self._begin_time = []
        self.name = name
        self.auto_show = auto_show
        self.idx = 0
        self.times = OrderedDict()
        self.sub_name = None
        self.last = 0
        self.enter_size = 0

    def __enter__(self,name):
        self.enter_size = len(self._begin_time)
        self.begin_time(name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time()
        while len(self._begin_time)>self.enter_size:
            self.end_time()

    
    def begin_time(self,name):
        torch.cuda.synchronize()
        if len(self._begin_time) > 0:
            name = self._begin_time[-1][0]+"."+name
        self.sub_name = name
        self._begin_time.append((name,time.time()))

    def end_time(self,*,new_time=None):
        if len(self._begin_time)==0:
            print(f"Haven't begin time.")
            return
        torch.cuda.synchronize()
        end_time = time.time()
        name,begin_time = self._begin_time[-1]
        if self.name is not None and len(self.name)>0:
            name = self.name+": "+name
        self.last = end_time-begin_time
        self.times[name] = self.last
        if self.auto_show:
            te = (end_time-begin_time)*1000
            fps = 1000/(te+1e-8)
            print(f"{name}: total time {te:.3f}ms, FPS={fps:.3f}.")
        self._begin_time = self._begin_time[:-1]

        if new_time is not None:
            self.begin_time(new_time)

    def end_all(self):
        while len(self._begin_time)>0:
            self.end_time()

    def reset(self,new_time=None):
        self.end_all()
        if new_time is not None:
            self.begin_time(new_time)

    def __repr__(self):
        res = "{\n"
        for k,v in self.times.items():
            res += f"{k}: {v}\n"
        res += "}"
        return res


class MovingAvg(object):
    def __init__(self,init_val = 0.0,momentum=0.99):
        self.v = init_val
        self.momentum = momentum

    def __call__(self, v):
        self.v = self.v*self.momentum+v*(1-self.momentum)

    def value(self):
        return self.v


class EstimateTimeCost(object):
    RECORD_STEP = 100
    def __init__(self,total_nr,auto_log=False,avg_step=2000):
        self.begin_time = None
        self.total_nr = total_nr
        self.process_nr = 0
        self.begin_step = 0
        self._time_datas = None
        self.t0 = None
        self.reset()
        self.auto_log = auto_log
        self.avg_step = avg_step

    def reset(self,total_nr = None):
        self.begin_time = time.time()
        self.t0 = time.time()
        if total_nr is not None:
            self.total_nr = total_nr
        self.process_nr = 0
        self.begin_step = 0
        self._time_datas = None

    def add_count(self):
        self.process_nr += 1
        if self.process_nr%EstimateTimeCost.RECORD_STEP == EstimateTimeCost.RECORD_STEP-1 and self._time_datas is None:
            self._time_datas = (self.process_nr,time.time())
        return self.__repr__()
    
    def set_process_nr(self,process_nr):
        self.process_nr = process_nr
        if self.process_nr%EstimateTimeCost.RECORD_STEP == EstimateTimeCost.RECORD_STEP-1 and self._time_datas is None:
            self._time_datas = (self.process_nr,time.time())
        return self.__repr__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        info = self.add_count()
        if self.auto_log:
            print(info)

    def time_elapse(self):
        return time.time() - self.t0

    def time_remaind(self):
        return ((time.time() - self.begin_time) / max(self.process_nr-self.begin_step, 1)) * (
                    self.total_nr- self.process_nr)
    
    def total_time(self):
        return ((time.time() - self.begin_time) / max(self.process_nr-self.begin_step, 1))*self.total_nr

    def __repr__(self):
        if self._time_datas is not None and self.process_nr-self._time_datas[0]>self.avg_step:
            self.begin_step = self._time_datas[0]
            self.begin_time = self._time_datas[1]
            self._time_datas = None

        left_time = ((time.time() - self.begin_time) / max(self.process_nr-self.begin_step, 1)) * (
                    self.total_nr- self.process_nr)
        finish_time = datetime.datetime.now() + datetime.timedelta(seconds=left_time)
        cur_str = datetime.datetime.now().strftime("%d %H:%M")
        ft_str = finish_time.strftime("%d %H:%M")
        #res = f"used {(time.time() - self.begin_time) / 3600:.3f}h, {left_time / 3600:.3f}h left, exp finish {str(d)}"
        #res = f"{(time.time() - self.begin_time) / 3600:.3f}h/{left_time / 3600:.3f}h/{str(d)}"
        res = f"{(time.time() - self.t0) / 3600:.3f}h/{left_time / 3600:.3f}h/{cur_str}/{ft_str}"
        return res

def time_this(func):
    @wraps(func)
    def wraps_func(*args,**kwargs):
        begin_t = time.time()
        res = func(*args,**kwargs)
        print(f"Time cost {time.time()-begin_t}s.")
        return res
    return wraps_func

def no_throw(func):
    @wraps(func)
    def wraps_func(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except Exception as e:
            print(f"ERROR: in no_throw wraps {e}")
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

def is_method_overridden(method, base_class, derived_class):
    """Check if a method of base class is overridden in derived class.

    Args:
        method (str): the method name to check.
        base_class (type): the class of the base class.
        derived_class (type | Any): the class or instance of the derived class.
    """
    assert isinstance(base_class, type), \
        "base_class doesn't accept instance, Please pass class instead."

    if not isinstance(derived_class, type):
        derived_class = derived_class.__class__

    base_method = getattr(base_class, method)
    derived_method = getattr(derived_class, method)
    return derived_method != base_method

def add_dict(lhv,rhv):
    res = dict(lhv)
    for k,v in rhv.items():
        if k in res:
            res[k] = res[k]+v
        else:
            res[k] = v
    return res

def sleep_for(hours):
    finish_time = datetime.datetime.now() + datetime.timedelta(seconds=int(hours*3600))
    sleep_until(finish_time)

def sleep_until(runtime):
    if isinstance(runtime,(str,bytes)):
        target_datetime = datetime.datetime.strptime(runtime, "%Y-%m-%d %H:%M:%S")
    else:
        target_datetime = runtime

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

def nparray(data,default_shape=[0],dtype=np.float32):
    res = np.array(data)
    if res.size == 0:
        return np.zeros(default_shape,dtype=dtype)
    return res

def is_int(v,eps=1e-6):
    return math.fabs(v-int(v))<eps

def to_fraction(v):
    '''
    将小数转化为分数
    example:
    v=0.4:
    return:
    2,5
    '''
    max_try = 20
    if is_int(v):
        return v
    denominator = 1
    for i in range(max_try):
        v *= 10
        denominator *= 10
        if is_int(v):
            break
    if not is_int(v):
        return v,denominator
    
    v = int(v)
    for _ in range(max_try):
        is_find = False
        for i in range(2,v+1):
            if v%i == 0 and denominator%i==0:
                v = v//i
                denominator = denominator//i
                is_find = True
                break
        if not is_find:
            break
    return v,denominator

def is_divide_exactly(lhv,rhv):
    return (lhv//rhv)*rhv==lhv

def lowest_common_multiple(a,b):
    '''
    最小公倍数
    '''
    if a==b:
        return a

    max_v = max(a,b)
    min_v = min(a,b)

    if is_divide_exactly(max_v,min_v):
        return max_v

    res = max_v*min_v

    while True:
        for i in range(1,min_v):
            if is_divide_exactly(res,i):
                tmp_v = res//i
                if is_divide_exactly(tmp_v,max_v) and is_divide_exactly(tmp_v,min_v):
                    res = tmp_v
                    break
        if i>=min_v-1:
            break
    
    return res

def dump2file(obj,file):
    if isinstance(file,(str,bytes)):
        with open(file,"wb") as f:
            return pickle.dump(obj,f)
    else:
        return pickle.dump(obj,file)

def load_from_file(file):
    if isinstance(file,(str,bytes)):
        with open(file,"rb") as f:
            return pickle.load(f)
    else:
        return pickle.load(file)

def parse_version(version="0.0.0") -> tuple:
    """
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version. This
    function replaces deprecated 'pkg_resources.parse_version(v)'.

    Args:
        version (str): Version string, i.e. '2.0.1+cpu'

    Returns:
        (tuple): Tuple of integers representing the numeric part of the version and the extra string, i.e. (2, 0, 1)
    """
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))  # '2.0.1+cpu' -> (2, 0, 1)
    except Exception as e:
        print(f"WARNING ⚠️ failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0

def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    """
    Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        name (str, optional): Name to be used in warning message.
        hard (bool, optional): If True, raise an AssertionError if the requirement is not met.
        verbose (bool, optional): If True, print warning message if requirement is not met.
        msg (str, optional): Extra message to display if verbose.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Example:
        ```python
        # Check if current version is exactly 22.04
        check_version(current="22.04", required="==22.04")

        # Check if current version is greater than or equal to 22.04
        check_version(current="22.10", required="22.04")  # assumes '>=' inequality if none passed

        # Check if current version is less than or equal to 22.04
        check_version(current="22.04", required="<=22.04")

        # Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current="21.10", required=">20.04,<22.04")
        ```
    """
    if not current:  # if current is '' or None
        print(f"WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():  # current is package name rather than version string, i.e. current='ultralytics'
        try:
            name = current  # assigned package name to 'name' arg
            current = metadata.version(current)  # get version string from package name
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(f"WARNING ⚠️ {current} package is required but not installed") from e
            else:
                return False

    if not required:  # if required is '' or None
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  # split '>=22.04' -> ('>=', '22.04')
        if not op:
            op = ">="  # assume >= if no op passed
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op == ">=" and not (c >= v):
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning = f"WARNING ⚠️ {name}{op}{version} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(warning)  # assert version requirements met
        if verbose:
            print(warning)
    return result




def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Default: 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert 'parrots' not in version_str
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])  # type: ignore
    else:
        release.extend([0, 0])
    return tuple(release)

