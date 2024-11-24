import imageio.v2 as imageio
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import json
from skimage.transform import resize

class FitLandmark(nn.Module):
    def __init__(self, input_img_path, target_body_img_path, hair_seg_path, source_lmk_path, target_lmk_path, width=512):
        super(FitLandmark, self).__init__()

        self.source_lmk = torch.from_numpy(np.load(source_lmk_path)[:, :2]).float().cuda().unsqueeze(0)
        with open(target_lmk_path,'r') as f:
            target_lmk = json.load(f)
        self.target_lmk = torch.from_numpy(np.array(target_lmk)).float().cuda().unsqueeze(0)

        self.input_img = np.array(imageio.imread(input_img_path)) #np.array()*255.0

        self.target_body_img = np.array(imageio.imread(target_body_img_path))
        self.hair_seg = np.array(imageio.imread(hair_seg_path))

        self.width = width

        self.lmk_loss = nn.MSELoss()

        #compute initial value
        source_scale = self.source_lmk[0,9,1] - self.source_lmk[0,28,1]
        source_center_x = self.source_lmk[0, :27, 0].mean().item()
        source_center_y = self.source_lmk[0, :27, 1].mean().item()

        target_scale = self.target_lmk[0,9,1] - self.target_lmk[0,28,1]
        target_center_x = self.target_lmk[0, :27, 0].mean().item()
        target_center_y = self.target_lmk[0, :27, 1].mean().item()

        init_scale = target_scale/source_scale
        init_trans_x = target_center_x - source_center_x
        init_trans_y = target_center_y - source_center_y

        self.register_buffer('scale_base', nn.Parameter(torch.Tensor([init_scale])))
        self.register_buffer('translate_base', nn.Parameter(torch.Tensor([[[init_trans_x, init_trans_y]]])))

        self.register_parameter('scale_dis', nn.Parameter(torch.zeros_like(self.scale_base)))
        self.register_parameter('translate_dis', nn.Parameter(torch.zeros_like(self.translate_base)))
        # print(self.translate.shape)

    
    def get_img_lmk(self):
        gt_lmk_coor = np.floor(self.target_lmk.detach().cpu().numpy()[0]).astype(np.int32)
        init_lmk_coor = np.floor(self.source_lmk.detach().cpu().numpy()[0]).astype(np.int32)
        pred_lmk_coor = np.floor(self.get_shiftted_lmk().detach().cpu().numpy()[0]).astype(np.int32)

        gt_lmk_coor = np.clip(gt_lmk_coor,0,511)
        pred_lmk_coor = np.clip(pred_lmk_coor,0,511)

        output_img = self.target_body_img.copy()
        output_img[gt_lmk_coor[:,1], gt_lmk_coor[:,0], :4] = [255,0,0, 255]
        output_img[pred_lmk_coor[:,1], pred_lmk_coor[:,0], :4] = [0,0,255, 255]
        output_img[init_lmk_coor[:,1], init_lmk_coor[:,0], :4] = [0,255,0, 255]

        return output_img.astype(np.uint8)
    
    def composite_img(self, is_shiftdown=False):
        output_img = self.target_body_img.copy()

        #resize source image
        target_width = int(self.width*(self.scale_base + self.scale_dis).item())
        # print(target_width)
        resized_input_img = resize(self.input_img.copy()/255.0, (target_width, target_width))*255.0
        resized_hair_seg = resize(self.hair_seg.copy()/255.0, (target_width, target_width))*255.0

        #compute idxs of resized source
        hair_seg = resized_hair_seg > (255*0.1)
        hair_seg_idx = np.array(np.where(hair_seg)).T #n*2

        hair_seg_idx_source = hair_seg_idx #(hair_seg_idx*self.scale.item())

        trans = (self.translate_base + self.translate_dis)[0].detach().cpu().numpy()
        if is_shiftdown:
            trans[:, 1] = trans[:, 1] + 10

        invert_trans = trans[:,::-1]
        # hair_seg_idx_target = (hair_seg_idx*self.scale.item() + invert_trans)
        hair_seg_idx_target = (hair_seg_idx + invert_trans)


        for i in range(hair_seg_idx.shape[0]):
            output_img[int(hair_seg_idx_target[i,0]), int(hair_seg_idx_target[i,1]), :3] = resized_input_img[int(hair_seg_idx_source[i,0]), int(hair_seg_idx_source[i,1]), :3]
            output_img[int(hair_seg_idx_target[i,0]), int(hair_seg_idx_target[i,1]), 3] = 255 #change alpha

        return output_img.astype(np.uint8)

    def get_shiftted_lmk(self):
        return self.source_lmk * (self.scale_base + self.scale_dis) + (self.translate_base + self.translate_dis)

    def forward(self):
        shiftted_lmk = self.get_shiftted_lmk()

        # loss = torch.sum(torch.abs(shiftted_lmk - self.target_lmk))
        loss = torch.sum(torch.abs(shiftted_lmk[:,:27,:] - self.target_lmk[:,:27,:]))

        return loss

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def opt(sample_info):
    target_body_img_path = './data/alignment/front.png'
    target_lmk_path = './data/alignment/front_lmk.json'

    input_img_path = sample_info[0]
    hair_seg_path = sample_info[1]
    source_lmk_path = sample_info[2]
    output_img = sample_info[3]
    vis_lmk_fit = sample_info[4]

    lr = 0.1
    num_epoch = 5001
    schedule = [2500]

    # load mesh and transform to camera space
    lmk_opt = FitLandmark(input_img_path, target_body_img_path, hair_seg_path, source_lmk_path, target_lmk_path).cuda()

    optimizer = torch.optim.Adam(lmk_opt.parameters(), lr, betas=(0.5, 0.99))

    #loop = tqdm.tqdm(list(range(0, num_epoch)))
    loop = list(range(0, num_epoch))

    for i in loop:

        loss = lmk_opt.forward()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = adjust_learning_rate(optimizer, i, lr, schedule, 0.5)

        if i == (num_epoch-1):
            image = lmk_opt.get_img_lmk()
            imageio.imwrite(vis_lmk_fit, image)

            image = lmk_opt.composite_img(is_shiftdown=True)
            # image = lmk_opt.composite_img(is_shiftdown=False)
            imageio.imwrite(output_img, image)
            

if __name__ == '__main__':

    img_dir = os.path.join('./data/alignment/', 'resized_img')
    seg_dir = os.path.join('./data/alignment/', 'seg')
    lmk_dir = os.path.join('./data/alignment/', 'lmk')
    output_dir = os.path.join('./data/alignment/', 'aligned_img')
    lmk_vis_dir = os.path.join('./data/alignment/', 'lmk_proj')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(lmk_vis_dir, exist_ok=True)

    items = os.listdir(seg_dir)

    sample_info = [(img_dir +'/'+ item, seg_dir +'/'+ item, 
                    lmk_dir +'/'+ item[:-3]+'npy', 
                    output_dir + '/' + item[:-3]+'png',  lmk_vis_dir +'/'+ item[:-3]+'png') for item in items]
    print('align hair with template body')
    for sample in tqdm(sample_info):
        try:
            opt(sample)
        except:
            print('Error in processing ', sample[0])
