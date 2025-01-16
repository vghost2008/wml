import os
import wml.wml_utils as wmlu
import argparse
import json
import shutil
import sys

'''
删除labelme标注中的imageData字段, 用于减小文件大小
'''

def parse_args():
    parser = argparse.ArgumentParser(description="remove labelme imagedata")
    parser.add_argument("src_dir",type=str,help="src dir")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    files = wmlu.get_files(args.src_dir,suffix=".json")
    KEY = "imageData"
    for i,file in enumerate(files):
        sys.stdout.write(f"Process {i}/{len(files)}       \r")
        with open(file,"r",encoding="gb18030") as f:
            data = json.load(f)
        if data.get(KEY,None) is not None:
            save_path = file+".bk"
            shutil.move(file,save_path)
            data[KEY] = None
            with open(file,"w") as f:
                json.dump(data,f)
    print(f"find {args.src_dir} -name \"*.bk\" "+"-exec rm {} ';' ")
