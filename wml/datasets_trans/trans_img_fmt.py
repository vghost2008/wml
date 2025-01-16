import wml.img_utils as wmli
import argparse
import os.path as osp
import os
import wml.wml_utils as wmlu

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, help='source imgs directory')
    parser.add_argument('out_dir', type=str, help='output imgs directory')
    parser.add_argument('out_ext', type=str, help='output img fmt')
    parser.add_argument(
        '--ext',
        type=str,
        default='.jpg;;.bmp;;.jpeg;;.png;;.tif',
        #choices=['avi', 'mp4', 'webm','MOV'],
        help='video file extensions')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    files = wmlu.get_files(args.src_dir,suffix=args.ext)
    print(files)
    wmlu.create_empty_dir_remove_if(args.out_dir)
    for f in files:
        rp = wmlu.get_relative_path(f,args.src_dir)
        op = osp.join(args.out_dir,rp)
        op = wmlu.change_suffix(op,suffix=args.out_ext)
        try:
            os.makedirs(osp.dirname(op),exist_ok=True)
            wmli.read_and_write_img(f,op)
        except Exception as e:
            print(f"Trans {f} faild, {e}")
        print(f,"-->",op)
