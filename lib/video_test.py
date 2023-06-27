
import sys

import cv2

sys.path.insert(0,'..')
from os.path import abspath, dirname

current_path = dirname(dirname(abspath(__file__)))
sys.path.insert(0, current_path)
import argparse
import math
import warnings

import numpy as np
import torch
import torch.nn.parallel
import torchvision.transforms as transforms
from align import getTform_scale
from mobilenetv2 import mobilenetv2
from networks import Pip_mbnetv2, Pip_mbnetv2_precess_in
from PIL import Image

from FaceBoxesV2.faceboxes_detector import *

warnings.filterwarnings('ignore')

def get_meanface(meanface_file, num_nb, scale=1.0):
    with open(meanface_file, 'r', encoding='utf-8') as f:
        meanface = f.readlines()[0]

    meanface = meanface.strip().split()
    meanface = [float(x) for x in meanface]
    meanface = np.array(meanface).reshape(-1, 2)
    meanface = (meanface - 0.5) * scale + 0.5 
    meanface_indices = []
    for i in range(meanface.shape[0]):
        pt = meanface[i,:] 
        dists = np.sum(np.power(pt-meanface, 2), axis=1)
        indices = np.argsort(dists)
        meanface_indices.append(indices[1:1+num_nb])

    meanface_indices_reversed = {}
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i] = [[],[]]
    for i in range(meanface.shape[0]):
        for j in range(num_nb):
            meanface_indices_reversed[meanface_indices[i][j]][0].append(i)
            meanface_indices_reversed[meanface_indices[i][j]][1].append(j)

    max_len = 0
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        if tmp_len > max_len:
            max_len = tmp_len

    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        meanface_indices_reversed[i][0] += meanface_indices_reversed[i][0]*10
        meanface_indices_reversed[i][1] += meanface_indices_reversed[i][1]*10
        meanface_indices_reversed[i][0] = meanface_indices_reversed[i][0][:max_len]
        meanface_indices_reversed[i][1] = meanface_indices_reversed[i][1][:max_len]

    reverse_index1 = []
    reverse_index2 = []
    for i in range(meanface.shape[0]):
        reverse_index1 += meanface_indices_reversed[i][0]
        reverse_index2 += meanface_indices_reversed[i][1]
    return meanface_indices, reverse_index1, reverse_index2, max_len

def demo_image_mbnet(image,points5,net,preprocess, input_size,device,num_lms):
    net.eval()
    if points5 is None:
        det_crop = image.copy()
        det_crop = cv2.resize(det_crop, (input_size, input_size))
        inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
        inputs = preprocess(inputs).unsqueeze(0)
        inputs = inputs.to(device)

        outputs = net(inputs)
        lms_pred_merge = outputs.flatten()
        lms_pred_merge = lms_pred_merge.cpu().detach().numpy()
        
        return lms_pred_merge

    else:
        tform_landmark = getTform_scale(points5, input_size, warp_type="DL_LANDMARK")
        det_crop = image.copy()
        det_crop = cv2.warpAffine(det_crop, tform_landmark, (input_size, input_size))
        inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
        inputs = preprocess(inputs).unsqueeze(0)
        inputs = inputs.to(device)

        outputs = net(inputs)
        lms_pred_merge = outputs.flatten()
        lms_pred_merge = lms_pred_merge.cpu().detach().numpy()
        lms_pred_merge = lms_pred_merge * input_size
        M = np.concatenate((tform_landmark, np.matrix([0,0,1])))
        lmks = np.concatenate((lms_pred_merge.reshape(-1,2), np.ones((num_lms,1))), axis=1)
        lms_pred_noalign = np.dot(np.linalg.inv(M), lmks.T).T[:,:2]

        return lms_pred_noalign

class OneEuroFilter:
    def __init__(self,t0,x0,dx0=0.0,min_cutoff=0.001,beta=0.7,d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def smoothing_factor(self,t_0,cutoff):
        r = 2 * math.pi * cutoff * t_0
        return r/(r+1)
    
    def exponential_smoothing(self,a,x,x_pred):
        return a * x + (1-a)*x_pred
    
    def __call__(self,t,x):
        t_e = t-self.t_prev
        a_d = self.smoothing_factor(t_e,self.d_cutoff)
        dx = (x-self.x_prev)/t_e
        dx_hat = self.exponential_smoothing(a_d,dx,self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e,cutoff)
        x_hat = self.exponential_smoothing(a,x,self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing_images configurations')
    parser.add_argument("--video_path",default="./002.avi", type=str)
    parser.add_argument("--num_nb",default=20, type=str)
    parser.add_argument("--width_mult",default=0.35, type=str)
    parser.add_argument("--num_lms",default=222, type=str)
    parser.add_argument("--input_size",default=192, type=str)
    parser.add_argument("--net_stride",default=32, type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    VIDEO_NAME = args.video_path
    RESULT_NAME = 'results_'+args.video_path.split("/")[-1]

    meanface_indices,reverse_index1,reverse_index2,max_len = get_meanface('./meanface_222.txt', args.num_nb)

    mbnet = mobilenetv2(width_mult=args.width_mult)
    net = Pip_mbnetv2(mbnet,args.num_nb,num_lms=args.num_lms,input_size=args.input_size,net_stride=args.net_stride)
    model_process_in = Pip_mbnetv2_precess_in(net, num_nb=args.num_nb,num_lms=args.num_lms,input_size=args.input_size
                                              ,net_stride=args.net_stride,reverse_index1=reverse_index1,
                                              reverse_index2=reverse_index2,max_len=max_len)
    model_CKPT = torch.load('./post_model.pth')
    model_process_in.load_state_dict(model_CKPT,strict=False)
    model_process_in = model_process_in.to(device)

    print("load model correct!")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])

    detector = FaceBoxesDetector('FaceBoxes', 'FaceBoxesV2/weights/FaceBoxesV2.pth', True, device)
    my_thresh = 0.9
    det_box_scale = 1.2

    videoCapture = cv2.VideoCapture(f"{VIDEO_NAME}")
    if (videoCapture.isOpened()== False): 
        print("Error opening video stream or file")
        exit(0)
    fps = int(videoCapture.get(cv2.CAP_PROP_FPS))
    frames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("fps:{0}, frames:{1}, height:{2}, width:{3}".format(fps,frames,height,width))

    output_path = f"{RESULT_NAME}"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowriter = cv2.VideoWriter(output_path,fourcc,fps,(width,height),True)
    landmark_5 = None
    cnt = 0
    for i in range(frames):
        
        success, frame = videoCapture.read()
        if(i % 1 == 0 and success == True):
            print("writing video: %.2f%%" % (float(100 * i / frames)))
            detections, _ = detector.detect(frame, my_thresh, 1) 

            if len(detections) != 0:

                det_xmin = detections[0][2]
                det_ymin = detections[0][3]
                det_width = detections[0][4]
                det_height = detections[0][5]
                det_xmax = det_xmin + det_width - 1
                det_ymax = det_ymin + det_height - 1

                det_xmin -= int(det_width * (det_box_scale-1)/2)
                det_ymin -= int(det_height * (det_box_scale-1)/2)
                det_xmax += int(det_width * (det_box_scale-1)/2)
                det_ymax += int(det_height * (det_box_scale-1)/2)
                det_xmin = max(det_xmin, 0)
                det_ymin = max(det_ymin, 0)
                det_xmax = min(det_xmax, width-1)
                det_ymax = min(det_ymax, height-1)
                det_width = det_xmax - det_xmin + 1
                det_height = det_ymax - det_ymin + 1
                face = frame[det_ymin:det_ymax, det_xmin:det_xmax, :]
                cv2.rectangle(frame, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)

                if cnt == 0:
                    landmark_pred = demo_image_mbnet(face,landmark_5,model_process_in,preprocess,input_size=args.input_size,device=device,num_lms=args.num_lms)
                    landmark_pred = np.array(landmark_pred).reshape(-1,2)

                    for j in range(0,args.num_lms):
                        landmark_pred[j,0] = landmark_pred[j,0] * det_width + det_xmin
                        landmark_pred[j,1] = landmark_pred[j,1] * det_height + det_ymin
                        cv2.circle(frame,(int(landmark_pred[j,0]),int(landmark_pred[j,1])),1,(0,0,255),-1)

                    new_landmark_5 = landmark_pred[[118,138,36,218,219]]
                else:
                    p_filter = OneEuroFilter(0,landmark_pred,dx0=0.0,min_cutoff= 0.001,beta=0.7,d_cutoff=1.0)
                    landmark_pred = demo_image_mbnet(frame,new_landmark_5,model_process_in,preprocess,input_size=args.input_size,device=device,num_lms=args.num_lms)
                    landmark_pred = np.array(landmark_pred).reshape(-1,2)
                    new_landmark_5 = landmark_pred[[118,138,36,218,219]]

                    filtered = p_filter(cnt,landmark_pred)
                    landmark_pred = filtered 
                
                    for j in range(0,args.num_lms):
                        x_pred = landmark_pred[j,0]
                        y_pred = landmark_pred[j,1]
                        cv2.circle(frame,(int(x_pred),int(y_pred)),3,(0,0,255),-1)

                videowriter.write(frame.astype(np.uint8))
            cnt+=1

    videoCapture.release()
    videowriter.release()

    print("FINISHED!")
