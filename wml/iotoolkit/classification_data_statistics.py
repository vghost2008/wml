import wml.wml_utils as wmlu

def labels_statistics(labels,name="",classes_names=None):
    counter = wmlu.Counter()
    for l in labels:
        if isinstance(l,(list,tuple)):
            l = l[0]
        counter.add(l)
    
    
    counter = list(counter.items())
    counter.sort(key=lambda x:x[1],reverse=True)
    classes = []
    for l,v in counter:
        classes.append(l)
    print(f"Total {len(classes)} classes")
    print(list(classes))
    if len(name)>0:
        print(name)
    for l,v in counter:
        if classes_names is not None:
            l = classes_names[l]
        print(f"{l:>32}: {v:<8},  {v*100.0/len(labels):>3.2f}%")

    print(f"Total {len(labels)} samples.")
