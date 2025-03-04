import os
import os.path as osp
import shutil
import sys
import socket
from functools import partial
from wml.basic_img_utils import BASE_IMG_SUFFIX
from wml.walgorithm import remove_non_ascii

def get_filenames_in_dir(dir_path,suffix=None,prefix=None):
    if suffix is not None:
        suffix = suffix.split(";;")
    if prefix is not None:
        prefix = prefix.split(";;")

    def check_file(filename):
        is_suffix_good = False
        is_prefix_good = False
        if suffix is not None:
            for s in suffix:
                if filename.endswith(s):
                    is_suffix_good = True
                    break
        else:
            is_suffix_good = True
        if prefix is not None:
            for s in prefix:
                if filename.startswith(s):
                    is_prefix_good = True
                    break
        else:
            is_prefix_good = True

        return is_prefix_good and is_suffix_good
    res=[]
    for dir_path,_,files in os.walk(dir_path):
        for file in files:
            if suffix is not None or prefix is not None:
                if check_file(file):
                    res.append(file)
            else:
                res.append(file)
    res.sort()
    return res

def get_filepath_in_dir(dir_path,suffix=None,prefix=None,sort=True):
    if suffix is not None:
        suffix = suffix.split(";;")
    if prefix is not None:
        prefix = prefix.split(";;")
    def check_file(filename):
        is_suffix_good = False
        is_prefix_good = False
        if suffix is not None:
            for s in suffix:
                if filename.endswith(s):
                    is_suffix_good = True
                    break
        else:
            is_suffix_good = True
        if prefix is not None:
            for s in prefix:
                if filename.startswith(s):
                    is_prefix_good = True
                    break
        else:
            is_prefix_good = True

        return is_prefix_good and is_suffix_good

    res=[]
    for file in os.listdir(dir_path):
        path = os.path.join(dir_path,file)
        if os.path.isdir(path):
            continue
        if suffix is not None or prefix is not None:
            if check_file(file):
                res.append(os.path.join(dir_path, file))
        else:
            res.append(os.path.join(dir_path,file))

    if sort:
        res.sort()

    return res

def find_files(dir_path,suffix=None,prefix=None,followlinks=False):
    '''
    suffix: example ".jpg;;.jpeg" , ignore case
    '''
    dir_path = os.path.expanduser(dir_path)

    if os.path.isfile(dir_path):
        return [dir_path]

    if suffix is not None:
        suffix = suffix.split(";;")
        suffix = [x.lower() for x in suffix]
    if prefix is not None:
        prefix = prefix.split(";;")
    def check_file(filename):
        is_suffix_good = False
        is_prefix_good = False
        if suffix is not None:
            for s in suffix:
                if filename.lower().endswith(s):
                    is_suffix_good = True
                    break
        else:
            is_suffix_good = True
        if prefix is not None:
            for s in prefix:
                if filename.startswith(s):
                    is_prefix_good = True
                    break
        else:
            is_prefix_good = True

        return is_prefix_good and is_suffix_good

    for dir_path,_,files in os.walk(dir_path,followlinks=followlinks):
        for file in files:
            if suffix is not None or prefix is not None:
                if check_file(file):
                    yield os.path.join(dir_path, file)
            else:
                yield os.path.join(dir_path,file)

def recurse_get_filepath_in_dir(dir_path,suffix=None,prefix=None,followlinks=False):
    '''
    suffix: example ".jpg;;.jpeg" , ignore case
    '''

    if isinstance(dir_path,(list,tuple)):
        return recurse_get_filepath_in_dirs(dir_path,suffix=suffix,prefix=prefix,followlinks=followlinks)

    dir_path = os.path.expanduser(dir_path)

    if os.path.isfile(dir_path):
        return [dir_path]

    if suffix is not None:
        suffix = suffix.split(";;")
        suffix = [x.lower() for x in suffix]
    if prefix is not None:
        prefix = prefix.split(";;")
    def check_file(filename):
        is_suffix_good = False
        is_prefix_good = False
        if suffix is not None:
            for s in suffix:
                if filename.lower().endswith(s):
                    is_suffix_good = True
                    break
        else:
            is_suffix_good = True
        if prefix is not None:
            for s in prefix:
                if filename.startswith(s):
                    is_prefix_good = True
                    break
        else:
            is_prefix_good = True

        return is_prefix_good and is_suffix_good

    res=[]
    for dir_path,_,files in os.walk(dir_path,followlinks=followlinks):
        for file in files:
            if suffix is not None or prefix is not None:
                if check_file(file):
                    res.append(os.path.join(dir_path, file))
            else:
                res.append(os.path.join(dir_path,file))
    res.sort()
    return res

get_files = recurse_get_filepath_in_dir
get_img_files = partial(recurse_get_filepath_in_dir,suffix=BASE_IMG_SUFFIX)

def recurse_get_subdir_in_dir(dir_path,predicte_fn=None,append_self=False):
    res=[]
    for root,dirs,_ in os.walk(dir_path):
        for dir in dirs:
            path = os.path.join(root,dir)
            if predicte_fn is not None:
                if not predicte_fn(path):
                    continue
            dir = path.replace(dir_path,"")
            if dir.startswith("/"):
                dir = dir[1:]
            res.append(dir)
    res.sort()
    if append_self:
        res.append("")
    return res

def get_subdir_in_dir(dir_path,sort=True,append_self=False,absolute_path=False):
    '''
    返回子目录名，如果absolute_path为True则返回子目录的绝对路径
    '''

    res=[]
    for file in os.listdir(dir_path):
        path = os.path.join(dir_path,file)
        if os.path.isdir(path):
            res.append(file)
    if append_self:
        res.append("")

    if sort:
        res.sort()

    if absolute_path:
        res = [os.path.abspath(os.path.join(dir_path,x)) for x in res]

    return res

def dir_path_of_file(file_path):
    return osp.dirname(osp.abspath(file_path))

def parent_dir_path_of_file(file_path):
    return osp.dirname(dir_path_of_file(file_path))

def sibling_file_path(file_path,sibling_name):
    dir_path = dir_path_of_file(file_path)
    return osp.join(dir_path,sibling_name)

def recurse_get_filepath_in_dirs(dirs_path,suffix=None,prefix=None,followlinks=False):
    files = []
    for dir in dirs_path:
        files.extend(recurse_get_filepath_in_dir(dir,suffix=suffix,prefix=prefix,followlinks=followlinks))
    files.sort()
    return files

def get_dirs(dir,subdirs):
    dirs=[]
    for sd in subdirs:
        dirs.append(os.path.join(dir,sd))
    return dirs

def try_link(src_file,dst_file):
    if os.path.isdir(dst_file):
        dst_file = os.path.join(dst_file,os.path.basename(src_file))
    try:
        os.link(src_file,dst_file)
    except Exception as e:
        print(f"{e} try copy file.")
        try:
            shutil.copy(src_file,dst_file)
        except Exception as e:
            print(f"{e} copy file {src_file} to {dst_file} faild.")

def copy_and_rename_file(input_dir,output_dir,input_suffix=".jpg",out_name_pattern="IMG_%04d.jpg",begin_index=1):
    inputfilenames = recurse_get_filepath_in_dir(input_dir,suffix=input_suffix)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    i = begin_index
    for file in inputfilenames:
        new_path = os.path.join(output_dir,out_name_pattern%(i))
        shutil.copyfile(file,new_path)
        print("copy %s to %s.\n"%(file,new_path))
        i = i+1
    print("Copy finish.")

def safe_copy(src_file,dst_file):
    if os.path.exists(dst_file) and os.path.isdir(dst_file):
        dst_file = os.path.join(dst_file,os.path.basename(src_file))
        safe_copy(src_file,dst_file)
        return


    r_dst_file = dst_file
    if os.path.exists(r_dst_file):
        r_base_name = base_name(dst_file)
        r_suffix = suffix(dst_file)
        dst_dir = os.path.dirname(dst_file)
        index = 1
        while os.path.exists(r_dst_file):
            r_dst_file = os.path.join(dst_dir,r_base_name+f"_{index:02}."+r_suffix)
            index += 1

    shutil.copy(src_file,r_dst_file)

def copy_file(src_file,dst_file):
    if os.path.exists(dst_file) and os.path.isdir(dst_file):
        dst_file = os.path.join(dst_file,os.path.basename(src_file))
        shutil.copy(src_file,dst_file)
        return
    shutil.copy(src_file,dst_file)

def base_name(v,process_suffix=True):
    if v[-1] == "/" or v[-1] == "\\":
        v = v[:-1]
    base_name = os.path.basename(v)

    if not process_suffix:
        return base_name

    index = base_name.rfind(".")
    if -1 == index:
        return base_name
    else:
        return base_name[:index]

def simple_base_name(v,process_suffix=True):
    name = base_name(v,process_suffix=process_suffix)
    name = remove_non_ascii(name)
    name = name.replace("\\","")
    return name

def simple_path(path):
    path = remove_non_ascii(path)
    path = path.replace("\\","")
    return path

def remove_path_spliter(v):
    if v[-1] == "/" or v[-1] == "\\":
        v = v[:-1]
    return v

def suffix(v):
    base_name = os.path.basename(v)
    index = base_name.rfind(".")
    if -1 == index:
        return base_name
    else:
        return base_name[index+1:]

def home_dir(sub_path=None):
    if sub_path is None:
        return os.path.expandvars('$HOME')
    else:
        return os.path.join(os.path.expandvars('$HOME'),sub_path)

def get_relative_path(path,ref_path):
    '''
    Example:
    path="/root/data/x.img", ref_path="/root"
    return:
    data/x.img
    '''

    if ref_path is None or path is None:
        return path

    path = osp.abspath(path)
    if isinstance(ref_path,(list,tuple)):
        ref_path = [osp.abspath(osp.expanduser(rp)) for rp in ref_path]
        for rp in ref_path:
            if path.startswith(rp):
                ref_path = rp
                break
        if isinstance(ref_path,(list,tuple)):
            return path
    else:
        ref_path = osp.abspath(ref_path)
        if not path.startswith(ref_path):
            return path
    if len(path)<=len(ref_path):
        return path
    res = path[len(ref_path):]
    if res[0] == osp.sep:
        return res[1:]
    return res


'''
suffix: suffix name without dot
'''
def change_suffix(path,suffix):
    dir_path = os.path.dirname(path)
    return os.path.join(dir_path,base_name(path)+"."+suffix)

def change_name(path,suffix=None,prefix=None,basename=None):
    dir_path = os.path.dirname(path)
    if basename is None:
        basename = base_name(path)
    if prefix is not None:
        basename = prefix+basename
    if suffix is not None:
        basename = basename+suffix
    fmt_suffix = os.path.splitext(path)[-1]
    return os.path.join(dir_path,basename+fmt_suffix)

def change_dirname(path,dir):
    basename = os.path.basename(path)
    return os.path.join(dir,basename)

def remove_hiden_file(files):
    res = []
    for file in files:
        if os.path.basename(file).startswith("."):
            continue
        res.append(file)

    return res

def safe_remove_dirs(dir_path,yes_to_all=False):
    if not os.path.exists(dir_path):
        return True
    if not yes_to_all:
        ans = input(f"Remove dirs in {dir_path} [y/N]?")
    else:
        ans = "y"
    if ans.lower() == "y":
        print(f"Remove dirs {dir_path}")
        shutil.rmtree(dir_path)
        return True
    else:
        return False

def create_empty_dir_remove_if(dir_path,key_word="tmp"):
    if key_word in dir_path:
        create_empty_dir(dir_path,True,True)
    else:
        create_empty_dir(dir_path,False,False)

def create_empty_dir(dir_path,remove_if_exists=True,yes_to_all=False):
    try:
        if remove_if_exists:
            if not safe_remove_dirs(dir_path,yes_to_all=yes_to_all):
                return False
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except:
        pass

    return True

def sync_data_dir(src_dir,dst_dir):
    if "vghost" in socket.gethostname():
        print("Skip sync data for vghost.")
        return
    print(f"Sync {src_dir} --> {dst_dir}")
    if not os.path.exists(src_dir):
        print(f"Error src dir {src_dir} dosen't exists.")
        return
    if not os.path.exists(dst_dir):
        print(f"Dst dir {dst_dir} dosen't exists, make dirs")
        os.makedirs(dst_dir)
    if not src_dir.endswith("/"):
        src_dir += "/"
    if not dst_dir.endswith("/"):
        dst_dir += "/"
    if src_dir == dst_dir:
        print(f"src dir and the dst dir is the same one {src_dir}, skip.")
        return
    src_dir += "*"
    cmd = f"cp -vup {src_dir} {dst_dir}"
    print(cmd)
    os.system(cmd)
    sys.stdout.flush()

def get_unused_path(path):
    if not os.path.exists(path):
        return path
    org_path = path
    if org_path[-1] == "/":
        org_path = org_path[:-1]
    i = 0
    while os.path.exists(path):
        path = org_path + f"_{i}"
        i += 1
    return path

def get_unused_path_with_suffix(path,begin_idx=0):
    if not os.path.exists(path):
        return path
    parts = osp.splitext(path)
    i = begin_idx
    while os.path.exists(path):
        path = parts[0]+ f"_{i}"+parts[1]
        i += 1
    return path

def make_dir_for_file(file_path):
    dir_name = osp.dirname(file_path)
    os.makedirs(dir_name,exist_ok=True)

def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)

def ls(path):
    sys.stdout.flush()
    os.system(f"ls -l {path}")
    sys.stdout.flush()
