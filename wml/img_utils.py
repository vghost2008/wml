#coding=utf-8
#import pydicom as dcm
import PIL.Image
from PIL import Image,ImageOps
import io
import scipy.misc
import matplotlib.image as mpimg
import numpy as np
import shutil
import os
import cv2
import copy
import random
import glob
import math
from collections import OrderedDict
from collections.abc import Iterable
from wml.basic_data_def import DEFAULT_COLOR_MAP as _DEFAULT_COLOR_MAP
from wml.basic_data_def import colors_tableau
import wml.object_detection2.basic_visualization as bodv
import torchvision
from wml.basic_img_utils import *

try:
    from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY, TurboJPEG
except ImportError:
    TJCS_RGB = TJPF_GRAY = TJPF_BGR = TurboJPEG = None
g_jpeg = None

'''def dcm_to_jpeg(input_file,output_file):
    ds = dcm.read_file(input_file)
    pix = ds.pixel_array
    scipy.misc.imsave(output_file, pix)
    return pix.shape'''

'''def dcms_to_jpegs(input_dir,output_dir):
    input_files = wmlu.recurse_get_filepath_in_dir(input_dir,suffix=".dcm")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for file in input_files:
        print('trans file \"%s\"'%(os.path.basename(file)))
        output_file = os.path.join(output_dir,wmlu.base_name(file)+".png")
        dcm_to_jpeg(file,output_file)'''

def to_jpeg(input_file,output_file):
    _input_file = input_file.lower()
    if _input_file.endswith(".jpg") or _input_file.endswith(".jpeg"):
        shutil.copyfile(input_file,output_file)
        #return None
    else:
        pix = mpimg.imread(input_file)
        scipy.misc.imsave(output_file, pix)
        #return pix.shape

def npgray_to_rgb(img):
    if img.ndim == 2:
        img = np.expand_dims(img,axis=2)
    shape = img.shape
    if shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)
    return img

def adjust_image_value_range(img):
    min = np.min(img)
    max = np.max(img)
    img = (img-min)*255.0/(max-min)
    return img

def npgray_to_rgbv2(img):
    img = adjust_image_value_range(img)
    def r(v):
        return np.where(v >= 127., 255., v * 255. / 127)
    def g(v):
        return (1. - np.abs(v - 127.) / 127.) * 255.
    def b(v):
        return np.where(v<=127.,255.,(1.-(v-127.)/127)*255.)
    if img.ndim == 2:
        img = np.expand_dims(img,axis=2)
    shape = img.shape
    if shape[2] == 1:
        img = np.concatenate([r(img), g(img), b(img)], axis=2)
    return img

def __nprgb_to_gray(img,keep_channels=False):
    img_gray = img * np.reshape(np.array([0.299, 0.587, 0.114], dtype=np.float32),[1,1,3])
    img_gray = np.sum(img_gray,axis=-1)
    if keep_channels:
        img_gray = np.stack([img_gray,img_gray,img_gray],axis=-1)
    return img_gray

def nprgb_to_gray(img,keep_channels=False,keep_dim=False):
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    if keep_channels:
        img_gray = np.stack([img_gray,img_gray,img_gray],axis=-1)
    elif keep_dim:
        img_gray = np.expand_dims(img_gray,axis=-1)
    return img_gray

def npbatch_rgb_to_gray(img,keep_channels=False):
    if not isinstance(img,np.ndarray):
        img = np.array(img)
    img_gray = img * np.array([0.299, 0.587, 0.114], dtype=np.float32)
    img_gray = np.sum(img_gray,axis=-1)
    if keep_channels:
        img_gray = np.stack([img_gray,img_gray,img_gray],axis=-1)
    return img_gray

def merge_image(src,dst,alpha):
    src = adjust_image_value_range(src)
    dst = adjust_image_value_range(dst)
    if len(dst.shape)<3:
        dst = np.expand_dims(dst,axis=2)
    if src.shape[2] != dst.shape[2]:
        if src.shape[2] == 1:
            src = npgray_to_rgb(src)
        if dst.shape[2] == 1:
            dst = npgray_to_rgb(dst)

    return src*(1.0-alpha)+dst*alpha

def merge_hotgraph_image(src,dst,alpha):
    if len(dst.shape)<3:
        dst = np.expand_dims(dst,axis=2)
    if src.shape[2] != dst.shape[2]:
        if src.shape[2] == 1:
            src = npgray_to_rgb(src)

    src = adjust_image_value_range(src)/255.
    dst = adjust_image_value_range(dst)/255.
    mean = np.mean(dst)
    rgb_dst = npgray_to_rgbv2(dst)/255.

    return np.where(dst>mean,src*(1.0-(2.*dst-1.)*alpha)+rgb_dst*(2.*dst-1.)*alpha,src)

'''def resize_img(img,size):

    image_shape = img.shape

    if size[0]==image_shape[0] and size[1]==image_shape[1]:
        return img

    h_scale = (float(size[0])+0.45)/float(image_shape[0])
    w_scale = (float(size[1])+0.45)/float(image_shape[1])
    if len(img.shape)==2:
        return scipy.ndimage.zoom(img, [h_scale, w_scale])
    else:
        return scipy.ndimage.zoom(img, [h_scale, w_scale,1])'''

def nprandom_crop(img,size):
    size = list(copy.deepcopy(size))
    x_begin = 0
    y_begin = 0
    if img.shape[0]>size[0]:
        y_begin = random.randint(0,img.shape[0]-size[0])
    else:
        size[0] = img.shape[0]
    if img.shape[1]>size[1]:
        x_begin = random.randint(0,img.shape[1]-size[1])
    else:
        size[1] = img.shape[1]

    rect = [y_begin,x_begin,y_begin+size[0],x_begin+size[1]]
    return sub_image(img,rect)

def imread(filepath):
    img = cv2.imread(filepath,cv2.IMREAD_COLOR)
    cv2.cvtColor(img,cv2.COLOR_BGR2RGB,img)
    return img

def hpimread(filepath):
    #return gpu_imread(filepath)
    if TurboJPEG is not None:
        global g_jpeg
        if g_jpeg is None:
            g_jpeg = TurboJPEG()
        if os.path.splitext(filepath)[1].lower() in [".jpg",".jpeg"]:
            with open(filepath,"rb") as f:
                return g_jpeg.decode(f.read(),TJCS_RGB)
    return imread(filepath)

def gpu_imread(filepath):
    if os.path.splitext(filepath)[1].lower() not in [".jpg",".jpeg"]:
        return imread(filepath)

    datas = torchvision.io.read_file(filepath)
    imgs = torchvision.io.decode_jpeg(datas,device="cuda")
    imgs = imgs.cpu().numpy()
    return imgs


def imsave(filename,img):
    imwrite(filename,img)

def imwrite(filename, img,size=None):
    '''
    size: (W,H)
    '''
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    dir_path = os.path.dirname(filename)
    if dir_path != "" and not os.path.exists(dir_path):
        os.makedirs(dir_path,exist_ok=True)
    img = np.ascontiguousarray(img)
    if len(img.shape)==3 and img.shape[2]==3:
        img = copy.deepcopy(img)
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR,img)
    if size is not None:
        img = resize_img(img,size=size,keep_aspect_ratio=True)
    cv2.imwrite(filename, img)

def read_and_write_img(src_path,dst_path):
    img = cv2.imread(src_path)
    cv2.imwrite(dst_path,img)

def imwrite_mask(filename,mask,color_map=_DEFAULT_COLOR_MAP):
    if os.path.splitext(filename)[1].lower() != ".png":
        print("WARNING: mask file need to be png format.")
    if not isinstance(mask,np.ndarray):
        mask = np.ndarray(mask)
    if len(mask.shape)==3:
        if mask.shape[-1] != 1:
            raise RuntimeError(f"ERROR mask shape {mask.shape}")
        mask = np.squeeze(mask,axis=-1)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(color_map)
    new_mask.save(filename)

def imwrite_mask_on_img(filename,img,mask,color_map=_DEFAULT_COLOR_MAP,ignored_label=255):
    r_img = bodv.draw_semantic_on_image(img,mask, color_map, ignored_label=ignored_label)
    imwrite(filename,r_img)

def imread_mask(filename):
    mask = Image.open(filename)
    return np.array(mask)

def videowrite(filename,imgs,fps=30,fmt="RGB"):
    if len(imgs)==0:
        return
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    write_size = imgs[0].shape[:2][::-1]
    video_writer = cv2.VideoWriter(filename, fourcc, fps,write_size)
    if fmt == "BGR":
        for img in imgs:
            video_writer.write(img)
    elif fmt=="RGB":
        for img in imgs:
            video_writer.write(img[...,::-1])
    else:
        print(f"ERROR fmt {fmt}.")
    video_writer.release()

class VideoWriter:
    def __init__(self,filename,fps=30,fmt='RGB'):
        self.video_writer = None
        self.fmt = fmt
        self.fps = fps
        self.filename = filename

    def __del__(self):
        self.release()

    def init_writer(self,img):
        if self.video_writer is not None:
            return
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        write_size = img.shape[:2][::-1]
        self.video_writer = cv2.VideoWriter(self.filename, fourcc, self.fps, write_size)

    def write(self,img):
        if self.video_writer is None:
            self.init_writer(img)
        fmt = self.fmt
        if fmt == "BGR":
            self.video_writer.write(img)
        elif fmt=="RGB":
            self.video_writer.write(img[...,::-1])
        else:
            print(f"ERROR fmt {fmt}.")

    def release(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

class VideoReader:
    def __init__(self,path,file_pattern="img_{:05d}.jpg",suffix=".jpg",preread_nr=0) -> None:
        if os.path.isdir(path):
            self.dir_path = path
            self.reader = None
            self.all_files = glob.glob(os.path.join(path,"*"+suffix))
            self.frames_nr = len(self.all_files)
            self.fps = 1
        else:
            self.reader = cv2.VideoCapture(path)
            self.dir_path = None
            self.frames_nr = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.reader.get(cv2.CAP_PROP_FPS)
            self.preread_nr = preread_nr
            if self.preread_nr>1:
                self.reader_buffer = OrderedDict()
            else:
                self.reader_buffer = None


        self.idx = 1
        self.file_pattern = file_pattern

    def __iter__(self):
        return self
    
    def __getitem__(self,idx):
        if self.dir_path is None:
            if self.preread_nr>1:
                if idx in self.reader_buffer:
                    return self.reader_buffer[idx]
                elif idx<self.idx-1:
                    raise NotImplemented()
                else:
                    for x in range(self.idx-1,idx+1):
                        if x in self.reader_buffer:
                            continue
                        ret,frame = self.reader.read()
                        if ret:
                            frame = frame[...,::-1]
                            self.reader_buffer[x] = frame
                    if idx in self.reader_buffer:
                        return self.reader_buffer[idx]

            raise NotImplemented()
        elif idx<self.frames_nr:
            if self.file_pattern is None:
                file_path = self.all_files[idx]
            else:
                file_path = os.path.join(self.dir_path,self.file_pattern.format(idx+1))
            img = cv2.imread(file_path)
            return img[...,::-1]
        else:
            raise RuntimeError()
    
    def __len__(self):
        if self.dir_path is not None:
            return self.frames_nr
        elif self.reader is not None:
            return self.frames_nr
        else:
            raise RuntimeError()

    def __next__(self):
        if self.reader is not None:
            if self.preread_nr>1:
                if self.idx-1 in self.reader_buffer:
                    frame = self.reader_buffer[self.idx-1]
                    ret = True
                else:
                    ret,frame = self.reader.read()
                    if not ret:
                        raise StopIteration()
                    frame = frame[...,::-1]
                    self.reader_buffer[self.idx-1] = frame
                    while len(self.reader_buffer)>self.preread_nr:
                        self.reader_buffer.popitem(last=False)
            else:
                retry_nr = 10
                while retry_nr>0:
                    ret,frame = self.reader.read()
                    retry_nr -= 1
                    if ret:
                        break

                if ret:
                    frame = frame[...,::-1]
            
            self.idx += 1
            if not ret:
                raise StopIteration()
            else:
                return frame
        else:
            if self.idx>self.frames_nr:
                raise StopIteration()
            if self.file_pattern is not None:
                file_path = os.path.join(self.dir_path,self.file_pattern.format(self.idx))
            else:
                file_path = self.all_files[self.idx-1]
            img = cv2.imread(file_path)
            self.idx += 1
            return img[...,::-1]

def imshow(winname,img):
    img = copy.deepcopy(img)
    cv2.cvtColor(img,cv2.COLOR_RGB2BGR,img)
    cv2.imshow(winname,img)

def np_resize_to_range(img,min_dimension,max_dimension=-1):
    new_shape = list(img.shape[:2])
    if img.shape[0]<img.shape[1]:
        new_shape[0] = min_dimension
        if max_dimension>0:
            new_shape[1] = min(int(new_shape[0]*img.shape[1]/img.shape[0]),max_dimension)
        else:
            new_shape[1] = int(new_shape[0]*img.shape[1]/img.shape[0])
    else:
        new_shape[1] = min_dimension
        if max_dimension>0:
            new_shape[0] = min(int(new_shape[1]*img.shape[0]/img.shape[1]),max_dimension)
        else:
            new_shape[0] = int(new_shape[1]*img.shape[0]/img.shape[1])

    return resize_img(img,new_shape)

def nppsnr(labels,predictions,max_v = 2):
    loss1 = np.mean(np.square(np.array(labels-predictions).astype(np.float32)))
    if loss1<1e-6:
        return 100.0
    return 10*np.log(max_v**2/loss1)/np.log(10)


class NPImagePatch(object):
    def __init__(self,patch_size):
        self.patch_size = patch_size
        self.patchs = None
        self.batch_size = None
        self.height = None
        self.width = None
        self.channel = None
    '''
    将图像[batch_size,height,width,channel]变换为[X,patch_size,patch_size,channel]
    '''
    def to_patch(self,images):
        patch_size = self.patch_size
        batch_size, height, width, channel = images.shape
        self.batch_size, self.height, self.width, self.channel = batch_size, height, width, channel
        net = np.reshape(images, [batch_size, height // patch_size, patch_size, width // patch_size, patch_size,
                                  channel])
        net = np.transpose(net, [0, 1, 3, 2, 4, 5])
        self.patchs = np.reshape(net, [-1, patch_size, patch_size, channel])
        return self.patchs

    def from_patch(self,patchs=None):
        assert self.patchs is not None,"Must call to_path first."
        if patchs is not None:
            self.patchs = patchs
        batch_size, height, width, channel = self.batch_size, self.height, self.width, self.channel
        patch_size = self.patch_size
        net = np.reshape(self.patchs, [batch_size, height // patch_size, width // patch_size, patch_size, patch_size,
                                       channel])
        net = np.transpose(net, [0, 1, 3, 2, 4, 5])
        net = np.reshape(net, [batch_size, height, width, channel])
        return net

'''
bboxes:[N,4],[ymin,xmin,ymax,xmax], absolute coordinate
'''
def remove_boxes_of_img(img,bboxes,default_value=[127,127,127]):
    if not isinstance(bboxes,np.ndarray):
        bboxes = np.array(bboxes)
    if bboxes.shape[0] == 0:
        return img
    ymin,xmin,ymax,xmax = np.transpose(bboxes)
    ymin = np.maximum(ymin,0)
    xmin = np.maximum(xmin,0)
    ymax = np.minimum(ymax,img.shape[0])
    xmax = np.minimum(xmax,img.shape[1])
    bboxes = np.stack([ymin,xmin,ymax,xmax],axis=1)

    for box in bboxes:
        img[box[0]:box[2], box[1]:box[3]] = default_value
    return img

def img_info(img):
    if len(img.shape) == 3 and img.shape[-1]>1:
        img = nprgb_to_gray(img)
    return np.std(img)


'''
img: np.ndarray, [H,W,3], RGB order
return:
bytes of jpeg string
'''
def encode_img(img,quality=95):
    if isinstance(img,str):
        img = imread(img)
    pil_image = PIL.Image.fromarray(img)
    output_io = io.BytesIO()
    pil_image.save(output_io, format='JPEG',quality=quality)
    return output_io.getvalue()

def _jpegflag(flag='color', channel_order='bgr'):
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'color':
        if channel_order == 'bgr':
            return TJPF_BGR
        elif channel_order == 'rgb':
            return TJCS_RGB
    elif flag == 'grayscale':
        return TJPF_GRAY
    else:
        raise ValueError('flag must be "color" or "grayscale"')

def decode_img(buffer,fmt='rgb'):
    if TurboJPEG is not None:
        global g_jpeg
        if g_jpeg is None:
            g_jpeg = TurboJPEG()
        if fmt == 'rgb':
            img = g_jpeg.decode(buffer,TJCS_RGB)
        elif fmt=='gray':
            img = g_jpeg.decode(buffer,TJPF_GRAY)
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        return img

    buff = io.BytesIO(buffer)
    img = PIL.Image.open(buff)

    if fmt=='rgb':
        img = pillow2array(img, 'color')
    else:
        img = pillow2array(img, 'grayscale')

    return img

def pillow2array(img,flag='color'):
    # Handle exif orientation tag
    #if flag in ['color', 'grayscale']:
        #img = ImageOps.exif_transpose(img)
    # If the image mode is not 'RGB', convert it to 'RGB' first.
    if flag in ['color', 'color_ignore_orientation']:
        if img.mode != 'RGB':
            if img.mode != 'LA':
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert('RGB')
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert('RGBA')
                img = Image.new('RGB', img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        array = np.array(img)
    elif flag in ['grayscale', 'grayscale_ignore_orientation']:
        img = img.convert('L')
        array = np.array(img)
    else:
        raise ValueError(
            'flag must be "color", "grayscale", "unchanged", '
            f'"color_ignore_orientation" or "grayscale_ignore_orientation"'
            f' but got {flag}')
    return array

def get_standard_color(idx):
    idx = idx%len(colors_tableau)
    return colors_tableau[idx]


def get_img_size(img_path):
    '''
    return: [H,W]
    '''
    if not os.path.exists(img_path):
        print(f"ERROR: img file {img_path} not exists.")
        return [0,0,3]
    else:
        with PIL.Image.open(img_path) as im:
            return list(im.size)[::-1]
        #return list(wmli.imread(img_path).shape)
