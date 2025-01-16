import wml.wml_utils as wmlu
import random

class DataUnit:
    MAX_IDXS_LEN = 100
    def __init__(self,data):
        if not isinstance(data,(list,tuple)):
            raise RuntimeError("Error data type")
        self.data = data
        self._idxs = []
    
    def make_idxs(self):
        nr = max(1,int(DataUnit.MAX_IDXS_LEN/len(self.data)))
        self._idxs = []
        for i in range(nr):
            idxs = self.make_one_idxs()
            self._idxs.extend(idxs)

    def make_one_idxs(self):
        idxs = list(range(len(self.data)))
        random.shuffle(idxs)
        return idxs


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def sample(self):
        if len(self._idxs) == 0:
            self.make_idxs()
        idx = self._idxs[-1]
        self._idxs = self._idxs[:-1]
        return self.data[idx]

    def __repr__(self):
        return type(self).__name__+f",{self.data}"
    

def make_data_unit(datas,total_nr=None,nr_per_unit=None):
    assert total_nr is None or nr_per_unit is None, "Error arguments"
    if total_nr is not None:
        if total_nr>=len(datas):
            return datas
        datas = wmlu.list_to_2dlistv2(datas,total_nr)
    else:
        if nr_per_unit<=1:
            return datas
        datas = wmlu.list_to_2dlist(datas,nr_per_unit)

    datas = [DataUnit(x) for x in datas]
    return datas

class DataList:
    def __init__(self,data) -> None:
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        if isinstance(data,DataUnit):
            data = data.sample()
        return data
