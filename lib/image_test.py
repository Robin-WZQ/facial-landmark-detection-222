import os
import sys
from tqdm import tqdm

import cv2

sys.path.insert(0,'..')
import argparse
import warnings

import numpy as np
import torch
import torch.nn.parallel
import torchvision.transforms as transforms
from mobilenetv2 import mobilenetv2
from PIL import Image
from networks import Pip_mbnetv2,Pip_mbnetv2_precess_in

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

def center_crop(image, crop_size):
    image_width = image.size[0]
    image_height = image.size[1]
    crop_width = crop_size[0]
    crop_height = crop_size[1]
    if image_width < crop_width or image_height < crop_height:
        new_width = max(image_width, crop_width)
        new_height = max(image_height, crop_height)
        new_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
        new_image.paste(image, (int((new_width-image_width)/2), int((new_height-image_height)/2)))
    else:
        new_image = image.copy()
        new_width = image_width
        new_height = image_height
    left = int(float(new_width - crop_width)/2.0)
    up = int(float(new_height - crop_height)/2.0)
    right = left + crop_width
    bottom = up + crop_height
    new_image = new_image.crop((left, up, right, bottom))

    return new_image

def demo_image_mbnetv2(image_file,net,preprocess, num_lms,input_size,device,output_file):
    net.eval()
    image = Image.open(image_file).convert('RGB')
    width,height = image.size[0],image.size[1]
    image = image.resize((input_size, input_size), Image.BICUBIC)

    inputs = preprocess(image).unsqueeze(0)
    inputs = inputs.to(device)
    outputs = net(inputs)
    lms_pred_merge = outputs.flatten()
    lms_pred_merge = lms_pred_merge.cpu().detach().numpy()

    # the points size
    size = int(input_size/128*3)

    cv_img = cv2.cvtColor(np.array(image),cv2.COLOR_BGR2RGB)
    for i in range(num_lms):
        x_pred = lms_pred_merge[i*2] * input_size * 6
        y_pred = lms_pred_merge[i*2+1] * input_size *6
        cv_img = cv2.resize(cv_img,(input_size*6,input_size*6))
        cv2.circle(cv_img, (int(x_pred),int(y_pred)),size,(0,0,255),-1)

    out = cv2.resize(cv_img,(width,height))
    cv2.imwrite(output_file,out)

def main():
    " Post Process "
    parser = argparse.ArgumentParser(description='Post process configurations')
    parser.add_argument("--img_path", default="./meanface.jpg",type=str)
    parser.add_argument("--checkpoint_path", default="./post_model.pth",type=str)
    parser.add_argument("--num_nb",default=20, type=str)
    parser.add_argument("--width_mult",default=0.35, type=str)
    parser.add_argument("--num_lms",default=222, type=str)
    parser.add_argument("--input_size",default=192, type=str)
    parser.add_argument("--net_stride",default=32, type=str)
    args = parser.parse_args()

    # meanface
    _,reverse_index1,reverse_index2,max_len = get_meanface('./meanface_222.txt', args.num_nb)

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load our model
    mbnet = mobilenetv2(width_mult=args.width_mult)
    net = Pip_mbnetv2(mbnet,args.num_nb,num_lms=args.num_lms,input_size=args.input_size,net_stride=args.net_stride)
    model_process_in = Pip_mbnetv2_precess_in(net, num_nb=args.num_nb,num_lms=args.num_lms,input_size=args.input_size
                                              ,net_stride=args.net_stride,reverse_index1=reverse_index1,
                                              reverse_index2=reverse_index2,max_len=max_len)
    model_CKPT = torch.load('./post_model.pth')
    model_process_in.load_state_dict(model_CKPT,strict=False)
    model_process_in = model_process_in.to(device)

    print("load_model_correct!")
    # ==============================================

    output_file = "result.jpg"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])

    
    demo_image_mbnetv2(args.img_path,model_process_in,preprocess,num_lms=args.num_lms,input_size=args.input_size,device=device,output_file=output_file)

    print("Done!")
    # ==============================================

if __name__=="__main__":
    main()
