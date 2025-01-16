import wml.object_detection2.bboxes as odb
import wml.img_utils as wmli

def write_yolo_txt(save_path,img_path,labels,bboxes):
    '''
    bboxes: [y0,x0,y1,x1]
    '''
    h,w = wmli.get_img_size(img_path)[:2]
    bboxes = odb.absolutely_boxes_to_relative_boxes(bboxes,width=w,height=h)
    bboxes = odb.npchangexyorder(bboxes) #(x0,y0,x1,y1)
    bboxes = odb.npto_cyxhw(bboxes) #->(cx,cy,w,h)
    with open(save_path,"w") as f:
        for l,bbox in zip(labels,bboxes):
            #每一行为label cx cy w h
            f.write(f"{l} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

def write_yoloseg_txt(save_path,img_path,labels,masks):
    '''
    masks:WPolygonMasks
    '''
    h,w = wmli.get_img_size(img_path)[:2]
    with open(save_path,"w") as f:
        for l,mask_items in zip(labels,masks):
            #每一行为label x0 y0 x1 y1 ...
            if len(mask_items.points) ==0:
                continue
            points = mask_items.points[0]

            if len(points)<3:
                continue
            info = f"{l} "

            for p in points:
                x = p[0]/w
                y = p[1]/h
                info += f"{x} {y} "
            info += "\n"
            f.write(info)