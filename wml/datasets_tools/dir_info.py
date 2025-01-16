from argparse import ArgumentParser
import os
import wml.wml_utils as wmlu

'''
统计文件夹中文件的信息（包含子目录)
'''

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dir', type=str, help='source video directory')
    args = parser.parse_args()
    return args

def print_dir_info(dir,blank=""):
    max_line_width = 108
    space_offset = "  "
    add_black = ""
    counter = wmlu.Counter()
    t_mark = "├──"
    s_mark = "│  "
    print(blank+t_mark+wmlu.base_name(dir))
    blank = blank+s_mark
    for file in os.listdir(dir):
        path = os.path.join(dir,file)
        if os.path.isdir(path):
            print_dir_info(path,blank+add_black)
        else:
            suffix = os.path.splitext(path)[-1]
            counter.add(suffix)
    info = add_black+t_mark+">"
    for k,v in counter.items():
        cur_info = f"{k}:{v}; "
        if info is None:
            info = add_black+t_mark+space_offset
        info = info + cur_info
        if len(info)>=max_line_width:
            print(blank+info)
            info = None
    if info is not None and len(info)>0 and len(counter)>0:
        print(blank+info)

if __name__ == "__main__":
    args = parse_args()
    print_dir_info(args.dir)

