import os
import cv2
import time
import tqdm
import numpy as np
import json, imageio
import torch
import torch.nn.functional as F
import rembg

from utils.cam_utils import orbit_camera, OrbitCamera
from utils.gs_renderer import Renderer, MiniCam
from utils.criterion import PerceptualLoss, ssim

class Main:
    def __init__(self, opt):
        self.opt = opt
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        self.p_loss_func = PerceptualLoss().cuda()
        

        # l1 loss
        self.save_prefix = 'l1_'
        self.p_loss_factor = 0
        self.ssim_factor = 0

        if self.opt.use_vgg and self.opt.use_ssim:
            self.save_prefix = 'l1_p_ssim_'
            self.p_loss_factor = 1#0.1#1
            self.ssim_factor = 1000#200
        elif self.opt.use_vgg:
            self.save_prefix = 'l1_p_'
            self.p_loss_factor = 1#0.1#1
        elif self.opt.use_ssim:
            self.save_prefix = 'l1_ssim_'
            self.ssim_factor = 1000#200

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

        # override if provide a checkpoint
        if self.opt.load is not None:
            print('load gaussian from: ', self.opt.load)
            self.renderer.initialize(self.opt.load)            
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.stage = 1

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # default camera
        if self.opt.mvdream or self.opt.imagedream:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(self.opt.elevation, 90, self.opt.radius)
        else:
            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading HairEnhancer...")
            from guidance.zero123_utils import Zero123
            self.guidance_zero123 = Zero123(self.device, model_key=self.opt.zero123_path, img_size=self.opt.ref_size)
            print(f"[INFO] loaded HairEnhancer!")

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        # prepare embeddings
        with torch.no_grad():
            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        flag_fix = False
        for _ in range(self.train_steps):
            flag_fix = not flag_fix

            if self.step%int(self.opt.iters/self.stage) == 0:
                self.update_targets(self.step//int(self.opt.iters/self.stage))
                self.p_loss_factor *= 100#100

                # if self.p_loss_factor>100:
                #     self.p_loss_factor=100

            if self.step==0:
                self.renderer.gaussians.reset_scale()

            self.step += 1
            
            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0

            ### known view
            if self.input_img_torch is not None and (not self.opt.imagedream):
                cur_cam = self.fixed_cam
                out = self.renderer.render(cur_cam)

                # rgb loss
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                img_loss = 10000 * F.l1_loss(image, self.input_img_torch)
                loss = loss + img_loss

                # mask loss
                mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                mask_loss = 1000 * F.l1_loss(mask, self.input_mask_torch)
                loss = loss + mask_loss

            ### novel view (manual batch)
            render_resolution = 512
            images = []
            poses = []
            vers, hors, radii = [], [], []

            #random
            range_render = range(self.opt.batch_size)

            if self.opt.batch_size>1:
                view_id_in_extra = np.random.randint(1, 180, self.opt.batch_size).tolist()
            else:
                 view_id_in_extra = [int(np.random.randint(1, 180, self.opt.batch_size))]

            for rand_view_i in range_render:
                radius = 0
                hor = self.loaded_hors[view_id_in_extra[rand_view_i]]
                ver = self.loaded_vers[view_id_in_extra[rand_view_i]]

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                poses.append(pose)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam, bg_color=bg_color)

                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)
                    
            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

            if self.enable_zero123:
                refined_images = self.refined_images[view_id_in_extra]
                refined_images = torch.from_numpy(refined_images).to(self.device)

                l1_loss = 10000 * self.opt.lambda_zero123 * F.l1_loss(images, refined_images)
                loss = loss + l1_loss

                p_loss = self.p_loss_func(images, refined_images) * self.p_loss_factor
                loss = loss + p_loss

                ssim_loss = (1.0 - ssim(images, refined_images)) * self.ssim_factor
                loss = loss + ssim_loss

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # densify and prune
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter and self.step <= self.opt.iters:
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)


                if (self.step-self.opt.density_start_iter) % self.opt.prune_interval == 0:
                    self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=50)

                if (self.step-self.opt.density_start_iter) % self.opt.densification_interval == 0:
                    self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
                    # self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=1.5, max_screen_size=1)

                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()
        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.step==self.opt.iters:
            self.update_targets(self.stage, need_diffusion=False, save=True)


    def update_targets(self, stage_id, need_diffusion=True, save=False):
        print('stage %d'%stage_id)

        strength = 0.0
        guide = 1.0

        # strength = 0.6
        # guide = 5.0


        print('update using strength %f'%strength)

        self.refined_images = []
        #load view params
        with open('./data/camera_pose_relative.json', 'r') as f:
            poses = json.load(f)
        loaded_vers = poses[0]
        loaded_hors = poses[1]

        self.loaded_vers = loaded_vers
        self.loaded_hors = loaded_hors

        # counter_num = 60 #60
        counter_num = 4 #60
        #render coarse imgs as noise
        for orbit_id in tqdm.tqdm(range(int(180/counter_num))):
            render_resolution = 512
            images = []
            refined_images = []
            vers, hors, radii = [], [], []
            
            for counter_id in range(counter_num):
                # render random view
                ver = loaded_vers[orbit_id*counter_num + counter_id]
                hor = loaded_hors[orbit_id*counter_num + counter_id]
                radius = 0
                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam, bg_color=bg_color)

                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)

                mask = (out["alpha"].unsqueeze(0).repeat(1,3,1,1)>0.5).float() # [1, 3, H, W] in [0, 1]

                if need_diffusion:
                    ref_img = image[:,:3,:,:]
                    self.guidance_zero123.get_img_embeds(ref_img)
                    refined_image = self.guidance_zero123.refine(image, [0], [0], [radius], steps=50, guidance_scale=guide, strength=strength, default_elevation=self.opt.elevation).float()
                    refined_image = refined_image*mask + (mask*(-1)+1)
                    refined_images.append(refined_image)
                    self.refined_images.append(refined_image.detach().cpu().numpy())

            images = torch.cat(images, dim=0)
            if need_diffusion:
                refined_images = torch.cat(refined_images, dim=0)

            if save:
                #save refined imgs
                for counter_id in range(counter_num):
                    # render random view
                    save_id = orbit_id*counter_num + counter_id + 1
                    out_img = images[counter_id].permute(1,2,0).detach().cpu().numpy()*255
                    
                    os.makedirs(self.opt.outdir + '/%sgs_enhance_render/'%self.save_prefix, exist_ok=True)
                    imageio.imwrite(self.opt.outdir + '/%sgs_enhance_render/%s.png'%(self.save_prefix, str(save_id).zfill(4)), out_img.astype(np.uint8))
                    
        if need_diffusion:
            self.refined_images = np.concatenate(self.refined_images, axis=0)

    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

    @torch.no_grad()
    def save_model(self):
        # self.opt.outdir = self.opt.outdir + '/' + self.opt.save_path
        os.makedirs(self.opt.outdir, exist_ok=True)

        path = os.path.join(self.opt.outdir, self.opt.save_path + '_enhance.ply')
        self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")

    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters + self.opt.extra_iter):
                self.train_step()
            # do a last prune
            # self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=None)

        # save
        self.save_model()
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    os.makedirs(opt.outdir+'/video', exist_ok=True)

    opt.save_path = str(opt.save_path)
    opt.outdir = opt.outdir + '/' + opt.save_path

    os.makedirs(opt.outdir, exist_ok=True)

    main = Main(opt)

    main.train(opt.iters)
