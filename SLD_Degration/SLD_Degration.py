# coding=gbk
import os
import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
import torchvision
from torch.nn import functional as F
from mmdetection.realesrgan_degration.utils.degradations import circular_lowpass_kernel, random_mixed_kernels,random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from mmdetection.realesrgan_degration.utils import file_client
from mmdetection.realesrgan_degration.utils.logger import get_root_logger
from mmdetection.realesrgan_degration.utils.img_util import imfrombytes,img2tensor
from mmdetection.realesrgan_degration.utils.img_process_util import filter2D,USMSharp
from mmdetection.realesrgan_degration.utils.transforms import augment,paired_random_crop
from mmdetection.realesrgan_degration.utils.DiffJPEG import DiffJPEG
import PIL.Image as Image
class Datadegration():
    def __init__(self,img_paths,gt_size=96):
        super(RealESRGANDatadegration, self).__init__()
        self.paths = img_paths
        self.scale = 1 #缩放比例
        self.gt_size = gt_size #输入图像的尺寸，需要h=w
        gpu_id = None #0
        device = None #
        if gpu_id:
            self.device = torch.device(
                f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        # the first degradation process
        self.resize_prob = [0.1, 0.5, 0.4]  # up, down, keep
        self.resize_range = [0.95, 1.05]
        self.gaussian_noise_prob = 0.1
        self.noise_range = [1, 4]
        self.poisson_scale_range = [0.05, 2]
        self.gray_noise_prob = 0.1
        self.jpeg_range = [30, 40]

        # the second degradation process
        self.second_blur_prob = 0.1
        self.resize_prob2 = [0.2, 0.2, 0.6]  # up, down, keep
        self.resize_range2 = [0.95, 1.05]
        self.gaussian_noise_prob2 = 0.1
        self.noise_range2 = [1, 3]
        self.poisson_scale_range2 = [0.05, 2]
        self.gray_noise_prob2 = 0.1
        self.jpeg_range2 = [30, 40]

        # blur settings for the first degradation
        self.blur_kernel_size = 5
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob =[0.45, 0.25, 0.12, 0.03, 0.12, 0.03]  # a list for each kernel probability
        self.sinc_prob = 0.1
        self.blur_sigma = [0.5, 1.5]
        self.betag_range = [0.5, 4] # betag used in generalized Gaussian blur kernels
        self.betap_range = [1, 2] # betap used in plateau blur kernels

        ## blur settings for the second degradation
        self.blur_kernel_size2 =5
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.sinc_prob2 = 0.1
        self.blur_sigma2 = [0.8, 1.2]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]

        # a final sinc filter
        self.final_sinc_prob = 0.4
        ############################
        self.use_hflip = False
        self.use_rot = False
        ################
        self.kernel_range = [2 * v + 1 for v in range(3, 5)] ## kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        self.file_client = None
        self.usm_sharpener = USMSharp().to(self.device)  # do usm sharpening
        self.jpeger = DiffJPEG(differentiable=False).to(self.device) # # simulate JPEG compression artifacts
    @torch.no_grad()
    def kernel(self):
        if self.file_client is None:
            self.file_client = file_client.FileClient()

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1

        img_gt = imfrombytes(img_bytes, float32=True)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.use_hflip, self.use_rot)

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = self.gt_size
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        # img_gt = img_gt.unsqueeze(4)
        # print(img_gt.shape)
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path}
        return return_d

    @torch.no_grad()
    def synthesis(self):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        # training data synthesis
        data = self.kernel()
        gt = data['gt'].to(self.device)
        self.gt = gt.unsqueeze(0)
        self.gt_usm = self.usm_sharpener(self.gt)
        self.kernel1 = data['kernel1'].to(self.device)
        self.kernel2 = data['kernel2'].to(self.device)
        self.sinc_kernel = data['sinc_kernel'].to(self.device)

        ori_h, ori_w = self.gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(self.gt_usm, self.kernel1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        # gray_noise_prob = self.gray_noise_prob
        # if np.random.uniform() < self.gaussian_noise_prob:
        #     out = random_add_gaussian_noise_pt(
        #         out, sigma_range=self.noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
        # else:
        #     out = random_add_poisson_noise_pt(
        #         out,
        #         scale_range=self.poisson_scale_range,
        #         gray_prob=gray_noise_prob,
        #         clip=True,
        #         rounds=False)
        # JPEG compression
        # jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range)
        # out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        # out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.second_blur_prob:
            out = filter2D(out, self.kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.resize_prob2)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / self.scale * scale), int(ori_w / self.scale * scale)),
            mode=mode)
        # add noise
        # gray_noise_prob = self.gray_noise_prob2
        # if np.random.uniform() < self.gaussian_noise_prob2:
        #     out = random_add_gaussian_noise_pt(
        #         out, sigma_range=self.noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
        # else:
        #     out = random_add_poisson_noise_pt(
        #         out,
        #         scale_range=self.poisson_scale_range2,
        #         gray_prob=gray_noise_prob,
        #         clip=True,
        #         rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
            out = filter2D(out, self.sinc_kernel)
            # JPEG compression
            # jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            # out = torch.clamp(out, 0, 1)
            # out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            # jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            # out = torch.clamp(out, 0, 1)
            # out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // self.scale, ori_w // self.scale), mode=mode)
            out = filter2D(out, self.sinc_kernel)

        # clamp and round
        self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # random crop
        gt_size = self.gt_size
        (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                             self.scale)

        # training pair pool
        #self._dequeue_and_enqueue()
        # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
        self.gt_usm = self.usm_sharpener(self.gt)
        lq = self.lq.contiguous()
        lq = lq.squeeze(0)
        # lq = lq.cpu().numpy()
        # #lq = 255 * (1.0 - lq)
        # lq = Image.fromarray(lq.astype(np.uint8), mode='RGB')
        return lq

if __name__ == '__main__':

    input_dir = '../mmdetection/data/images'
    output_dir = '../mmdetection/data/image_test'
    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        print(filename)
        L = H if H > W else W
        new_img = np.ones(shape=(L, L, 3)) * 255
        new_img[:H, :W] = img
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, new_img)
        t = Datadegration(output_path, gt_size=L)
        result = t.synthesis()

        new_img = np.ones(shape=(H,W,3))*255

        new_img = result[:3,:H,:W]

        torchvision.utils.save_image(new_img,output_path)