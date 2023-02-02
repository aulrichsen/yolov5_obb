# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import math
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression_obb, print_args, scale_coords, scale_polys, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import poly2rbox, rbox2poly

from utils.augmentations import letterbox


#@torch.no_grad()
class YoloV5_OBB():

    def __init__(self,
        weights=ROOT / 'weights/cow_obb_best_weights.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=(1024, 1024),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

        self.view_img = view_img
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres 
        self.classes = classes
        self.agnostic_nms = agnostic_nms 
        self.max_det = max_det
        self.line_thickness = line_thickness
        self.hide_labels=hide_labels,  # hide labels
        self.hide_conf=hide_conf,  # hide confidences 
        self.half = half

        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn)
        self.stride, self.names, self.pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Half
        half &= (self.pt or jit or engine) and self.device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if self.pt or jit:
            self.model.model.half() if half else self.model.model.float()

        # Run inference
        self.model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup


    def video_inference(self, source):
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        
        # Dataloader    
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        bs = 1  # batch_size

        for frame, (path, im, im0, vid_cap, s) in enumerate(dataset):
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            
            #print("frame:", frame)
            pred = self.inference(im)
            #print(im.shape, im0.shape)

            for frame_pred in pred:   # Iterate through each frame prediction
                
                if self.view_img:
                    
                    # Stream results
                    print(frame_pred.shape, im.shape, im0.shape)
                    im0 = self.annotate_img(im0, im, frame_pred)
                    im0 = self.resize_img(im0, scale=2)
                    cv2.imshow("Predictions", im0)
                    cv2.waitKey(1)  # 1 millisecond

    def resize_img(self, im0, scale=2):
        return cv2.resize(im0, (int(im0.shape[1]/scale), int(im0.shape[0]/scale)))
        

    def annotate_img(self, im0, im, frame_pred):
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        if len(frame_pred):
            pred_poly = rbox2poly(frame_pred[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
            # Rescale polys from img_size to im0 size
            # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            pred_poly = scale_polys(im.shape[2:], pred_poly, im0.shape)
            frame_pred = torch.cat((pred_poly, frame_pred[:, -2:]), dim=1) # (n, [poly conf cls])

            # Write results
            for *poly, conf, cls in reversed(frame_pred):
                c = int(cls)  # integer class
                label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                # annotator.box_label(xyxy, label, color=colors(c, True))
                annotator.poly_label(poly, label, color=colors(c, True))
                    
        # Stream results
        return annotator.result()


    def inference(self, im):    

        # Inference
        with torch.no_grad():
            pred = self.model(im, augment=False, visualize=False)

        # NMS
        # pred: list*(n, [xylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
        pred = non_max_suppression_obb(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, multi_label=True, max_det=self.max_det)

        #center, w, h, theta = pred[:2], pred[2:3],im0 pred[3:4], pred[4:5]
            
        return pred

    def get_centre_and_angle(self, im):
        '''
        returns list of 2d arrays, each list element is cow detections for a give frame (batch index)
        dim 0 is cows
        dim 1 is x, y, angle (radians)
        '''

        pred = self.inference(im)

        center_and_angle = []
        for frame_pred in pred:     # Iterate through the prediction batch        
            center, w, h, theta = frame_pred[:, :2], frame_pred[:, 2:3], frame_pred[:, 3:4], frame_pred[:, 4:5]

            #print(center)
            #print(theta)
            center_and_angle.append(torch.cat((center, theta), dim=1))

        return center_and_angle


    def load_image(self, img_file):
        '''
        im is preprocessed image
        im0 is original image loaded in with opencv
        '''
        
        im0 = cv2.imread(img_file)
        img = self.pre_process_image(im0)
        return img, im0

    def pre_process_image(self, img0):
        """
        Pre-process an image loaded in with opencv
        """
        
        # Padded resize
        im = letterbox(img0, self.imgsz, stride=self.stride, auto=self.pt)[0]

        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        return im

'''
            center, w, h, theta = detection[:2], detection[2:3], detection[3:4], detection[4:5]
            #Cos, Sin = torch.cos(theta), torch.sin(theta)
            theta = math.degrees(theta)
            if h > w:
                # Correct angle to be relative to head
                if theta > 0:
                    theta -=90
                else:
                    theta += 90

            print("center:", center)print(model.inference(im))

            print("theta:", theta)
'''

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/cow_obb_best_weights.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='/media/test/4d846cae-2315-4928-8d1b-ca6d3a61a3c6/DroneVehicle/val/raw/images/', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1024], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    model = YoloV5_OBB(**vars(opt))

    #model.video_inference(source=opt.source)

    # Get raw predictions on example image
    img_file = 'dataset/cow_obb_padded/test/images/501_20211010T140000z_orig.jpg'
    im, im0 = model.load_image(img_file)
    pred = model.inference(im)

    annotated_img = model.annotate_img(im0, im, pred[0])
    annotated_img = model.resize_img(annotated_img)
    cv2.imshow("Predictions:", annotated_img)
    cv2.waitKey(0)
    
    im, im0 = model.load_image(img_file)
    #pred = model.inference(im)

    

    # Get centre and angle and fix bounding box sizes (200 x 70)
    centre_and_angle = model.get_centre_and_angle(im)[0]
    pred = torch.zeros(centre_and_angle.shape[0], 5)
    pred[:, :2] = centre_and_angle[:, :2]
    pred[:, 2] = 200    # Box length
    pred[:, 3] = 70    # Box Width
    pred[:, 4] = centre_and_angle[:, 2]
    pred = [pred]

    annotated_img = model.annotate_img(im0, im, pred[0])
    annotated_img = model.resize_img(annotated_img)
    cv2.imshow("Center and Angle Predictions (Fixed Box Size):", annotated_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
