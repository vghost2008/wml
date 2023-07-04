import torch
import sys

def show_info(data,pre_key=None):
    for k,v in data.items():
        if pre_key is not None:
            nk = f"{pre_key}.{k}"
        else:
            nk = k
        if torch.is_tensor(v):
            print(f"{nk}:{v.shape}")
        elif isinstance(v,dict):
            show_info(v,nk)

def show_path_info(path):
    data = torch.load(path)
    show_info(data)

if __name__ == "__main__":
    show_path_info(sys.argv[1])
