def __get_resample_nr(labels,resample_parameters):
    nr = 1
    for l in labels:
        if l in resample_parameters:
            nr = max(nr,resample_parameters[l])
    return nr

def resample(files,labels,resample_parameters):
    '''
    files: list of files
    labels: list of labels
    resample_parameters: {labels:resample_nr}
    '''
    new_files = []
    for f,l in zip(files,labels):
        nr = __get_resample_nr(l,resample_parameters)
        if nr>1:
            new_files = new_files+[f]*nr
            print(f"Repeat {f} {nr} times.")
        elif nr==1:
            new_files.append(f)
    print(f"{len(files)} old files --> {len(new_files)} new files")

    return new_files
