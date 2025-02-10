import numpy as np

def get_scales(base,step,nr):
    '''
    base: base sacle, like (256,512)
    step: int or (int,int)
    nr: total scale nr
    '''
    res = []
    hnr = nr//2
    if not isinstance(step,(list,tuple)):
        step = (step,step)

    for i in range(-hnr,hnr):
        res.append((base[0]+i*step[0],base[1]+i*step[1]))
    
    print(res)

if __name__ == "__main__":
    get_scales((2048,2560),64,7)
