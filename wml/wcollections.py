import numpy as np
import random
import copy
from collections import abc


class ExperienceBuffer():
    def __init__(self, buffer_size = 100000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0: (len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        if not isinstance(self.buffer[0],np.ndarray):
            data = random.sample(self.buffer,size)
            data = list(zip(*data))
            return [np.array(list(x)) for x in data]
        else:
            return np.reshape(np.array(random.sample(self.buffer, size)), [size]+list(self.buffer[0].shape))


class CycleBuffer:
    def __init__(self,cap=5):
        self.cap = cap
        self.buffer = []
    def append(self,v):
        self.buffer.append(v)
        l = len(self.buffer)
        if l>self.cap:
            self.buffer = self.buffer[l-self.cap:]

    def __getitem__(self, slice):
        return self.buffer[slice]

    def __len__(self):
        return len(self.buffer)

class AlwaysNullObj(object):
    def __init__(self,*args,**kwargs):
        print(f"Construct a always null object")
        pass

    def __getattr__(self, item):
        return self

    def __setattr__(self, key, value):
        pass

    def __delattr__(self, item):
        pass

    def __call__(self, *args, **kwargs):
        return self


class MDict(dict):
    def __init__(self, *args, **kwargs):
        '''

        Args:
            *args:
            **kwargs:

            example 1:
            x = MDict(dtype=list)
            x[1].append('a')
            x[2].append('b')
            x[1].append('c')
            print(x)
            output:
            {1: ['a', 'c'], 2: ['b']}

            example 2:
            x = MDict(dvalue=[])
            x[1].append('a')
            x[2].append('b')
            x[1].append('c')
            print(x)
            output:
            {1: ['a', 'c'], 2: ['b']}
        '''
        self.default_type = None
        self.default_value = None
        if "dtype" in kwargs:
            self.default_type = kwargs.pop("dtype")
        elif "dvalue" in kwargs:
            self.default_value = kwargs.pop("dvalue")
        super().__init__(*args,**kwargs)

    @classmethod
    def from_dict(cls,data:dict,auto_dtype=True):
        x = data.values()
        assert len(x)>0, "error dict data"
        if auto_dtype:
            dtype = type(list(x)[0])
            ret = cls(dtype=dtype)
        else:
            ret = cls(dvalue=None)
        for k,v in data.items():
            ret[k] = v
        return ret

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        return self.__getitem__(key)

    def __call__(self,key):
        return self.__getitem__(key)

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        if key in self:
            return super().__getitem__(key)
        elif self.default_type is not None:
            super().__setitem__(key,self.default_type())
            return super().__getitem__(key)
        elif self.default_value is not None:
            super().__setitem__(key,self.default_value)
        return None

    def __delattr__(self, key):
        del self[key]


class Counter(dict):
    '''
    Example:
    counter = Counter()
    counter.add("a")
    counter.add("a")
    counter.add("b")
    print(counter):
    {
        a: 2
        b: 1
    }
    '''
    def add(self,key,nr=1):
        if key in self:
            self[key] += nr
        else:
            self[key] = nr
        return self[key]
    
    def total_size(self):
        return np.sum(list(self.values()))

class EDict(dict):
    '''
    只能添加键值，不允许更新
    '''
    def __setitem__(self,item,value):
        if item in self:
            raise RuntimeError(f"ERROR: key {item} already exists.")
        super().__setitem__(item,value)

def safe_update_dict(target_dict,src_dict,do_raise=True):
    duplicate_keys = []
    for k in src_dict.keys():
        if k in target_dict and target_dict[k] != src_dict[k]:
            duplicate_keys.append(k)
    

    if len(duplicate_keys)>0:
        if do_raise:
            raise RuntimeError(f"key {duplicate_keys} already in target dict, target dict keys {list(target_dict.keys())}")
        else:
            print(f"ERROR: key {duplicate_keys} already in target dict, target dict keys {list(target_dict.keys())}")

    target_dict.update(src_dict)

def trans_dict_key2lower(data):
    res = type(data)()
    for k,v in data.items():
        res[k.lower()] = v
    return res

def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True

def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)