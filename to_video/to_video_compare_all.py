import os
import cv2
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser(description="Training script parameters")
parser.add_argument('--split', type=str, default="")
parser.add_argument('--output', type=str, default="/data4/paul/code/Data/real_T0.8/5orbit-60-box-bs4")
parser.add_argument('--size', type=int, default=512)
args = parser.parse_args()


video_save_root = '%s/video/'%args.output

split = args.split
root = '%s/%s/'%(args.output, split)

os.makedirs(video_save_root, exist_ok=True)

imgs_path0 = '%s/%s/%s.png'%('./data/alignment', 'resized_img', split)
# imgs_path0 = '%s/%s/%s.png/'%('/data4/paul/code/HairStep/results/real_imgs', 'resized_img', split)
if not os.path.exists(imgs_path0):
    imgs_path0 = '%s/%s/%s.jpg'%('./data/alignment', 'resized_img', split)
    # imgs_path0 = '%s/%s/%s.jpg'%('/data4/paul/code/HairStep/results/real_imgs', 'resized_img', split)


# name='l1_gs_refine_render_stage_3'# only mse
# All_img_path=root+name
name='coarse_gaussian_render'# add p loss
All_img_path=root+name
imgs=[]
files = os.listdir(All_img_path)
files.sort()
imgs_path1 = [os.path.join(All_img_path, img) for img in files]

name='l1_p_gs_refine_render'# add p loss
All_img_path=root+name
imgs=[]
files = os.listdir(All_img_path)
files.sort()
imgs_path2 = [os.path.join(All_img_path, img) for img in files]

name='l1_p_gs_enhance_render' # add p loss and ssim
All_img_path=root+name
imgs=[]
files = os.listdir(All_img_path)
files.sort()
imgs_path3 = [os.path.join(All_img_path, img) for img in files]

size = (args.size*4,args.size)  # w*h

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
cap_fps = 15

try:
    video = cv2.VideoWriter('%scompare_%s.mp4'%(video_save_root, split), fourcc, cap_fps, size)
    print('%scompare_%s.mp4'%(video_save_root, split))
    for i in tqdm(range(len(imgs_path1)), desc='make_video', ncols=80, unit='img'):
        img0=cv2.imread(imgs_path0)
        img1=cv2.imread(imgs_path1[i])
        img2=cv2.imread(imgs_path2[i])
        img3=cv2.imread(imgs_path3[i])

        img = cv2.hconcat([img0, img1, img2, img3])

        video.write(img)
    video.release()
except:
    ffile = open('%s%s.txt'%(video_save_root,split), 'w')
    ffile.write('fail')