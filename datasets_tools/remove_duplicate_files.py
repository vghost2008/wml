import os
import wml_utils as wmlu
import argparse

#如果在exclude_dir和src_dir文件夹中同时出现，则从src_dir中删除

def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    parser.add_argument("exc_dir",type=str,help="src dir")
    parser.add_argument("--ext",type=str,default=".jpg",help="img ext")
    parser.add_argument('--basename', action='store_true',help='whether to compare by basename.')
    args = parser.parse_args()
    return args

def remove_file(file):
    #print(f"Remove {file}.")
    #return
    os.remove(file)
    img_file = wmlu.change_suffix(file,"jpg")
    if os.path.exists(img_file):
        print(f"Remove {img_file}")
        os.remove(img_file)

def relative_path(path,ref_path):
    return wmlu.get_relative_path(path,ref_path)

def basename(path,*args,**kwargs):
    return os.path.basename(path)

if __name__ == "__main__":
    args = parse_args()
    #要处理的文件夹
    src_dir = args.src_dir
    #如果在exclude_dir和src_dir文件夹中同时出现，则从src_dir中删除
    exclude_dir = args.exc_dir
    suffix = args.ext
    if args.basename:
        name_func = basename
    else:
        name_func = relative_path
    
    files0 = wmlu.recurse_get_filepath_in_dir(src_dir,suffix=suffix)
    files1 = wmlu.recurse_get_filepath_in_dir(exclude_dir,suffix=suffix)
    files1 = [name_func(file,exclude_dir) for file in files1]
    total_skip = 0
    total_remove = 0
    files_to_remove = []
    for file in files0:
        base_name = name_func(file,src_dir)
        if base_name not in files1:
            print(f"Skip {base_name}")
            total_skip += 1
        else:
            print(f"Remove {file}")
            total_remove += 1
            files_to_remove.append(file)
    
    print(f"Files need to remove {len(files_to_remove)}:")
    wmlu.show_list(files_to_remove)
    res = input(f"remove {len(files_to_remove)} [y/n]")
    if res != 'y':
        print(f"Cancel.")
        exit(0)

    for file in files_to_remove:
        remove_file(file)
    
    print(f"Total files in src dir {len(files0)}, total files in exclude dir {len(files1)}.")
    print(f"Total skip {total_skip}, total remove {total_remove}, total process {total_skip+total_remove}")