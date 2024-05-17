import torch
import sys

if __name__ == "__main__":

    path = sys.argv[1]
    spath = sys.argv[2]
    data = torch.load(path)
    new_data = data['state_dict']
    with open(spath,"wb") as f:
        torch.save(new_data,f)