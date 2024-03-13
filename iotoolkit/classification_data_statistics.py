import wml_utils as wmlu

def labels_statistics(labels,name=""):
    counter = wmlu.Counter()
    for l in labels:
        counter.add(l)
    
    counter = list(counter.items())
    counter.sort(key=lambda x:x[0])
    if len(name)>0:
        print(name)
    for l,v in counter:
        print(f"{l:>8}: {v:<8},  {v*100.0/len(labels):>3.2f}%")
