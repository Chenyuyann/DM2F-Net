# coding: utf-8
import os

import numpy as np
import torch
import time
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT, HAZERD_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model import DM2FNet, DM2FNet_woPhy
from datasets import SotsDataset, OHazeDataset, HazeRDDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from skimage.color import rgb2lab
from skimage.transform import resize
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
# exp_name = 'RESIDE_ITS'
# exp_name = 'O-Haze'
exp_name = 'HazeRD'

args = {
    # RESIDE
    # 'snapshot': 'iter_40000_loss_0.01230_lr_0.000000',
    'snapshot': 'iter_40000_loss_0.01225_lr_0.000000',
    # 'snapshot': 'iter_35000_loss_0.01244_lr_0.000077',

    # O-Haze
    # 'snapshot': 'iter_19000_loss_0.04261_lr_0.000014',
    # 'snapshot': 'iter_20000_loss_0.05028_lr_0.000000',
}

to_test = {
    # 'SOTS': TEST_SOTS_ROOT,
    # 'O-Haze': OHAZE_ROOT,
    'HazeRD': HAZERD_ROOT,
}

to_pil = transforms.ToPILImage()

def to_lab(image):
    """
    Convert an RGB image to LAB color space.
    """
    lab_image = rgb2lab(image)
    return lab_image

def compute_ciede2000_downsampled(res, gt, downsample_size=(100, 100)):
    # Resize images
    res_downsampled = resize(res, downsample_size, anti_aliasing=True)
    gt_downsampled = resize(gt, downsample_size, anti_aliasing=True)

    # Convert to LAB
    res_lab = rgb2lab(res_downsampled)
    gt_lab = rgb2lab(gt_downsampled)

    # Calculate CIEDE2000
    ciede2000_scores = []
    for y in range(downsample_size[0]):
        for x in range(downsample_size[1]):
            r_pixel_lab = LabColor(*res_lab[y, x])
            gt_pixel_lab = LabColor(*gt_lab[y, x])
            ciede2000 = delta_e_cie2000(r_pixel_lab, gt_pixel_lab)
            ciede2000_scores.append(ciede2000)

    # Return the average CIEDE2000 score
    return np.mean(ciede2000_scores)

def main():
    with torch.no_grad():
        criterion = nn.L1Loss().cuda()

        # 启动整体计时
        overall_start_time = time.time()

        for name, root in to_test.items():
            if 'SOTS' in name:
                net = DM2FNet().cuda()
                dataset = SotsDataset(root)
            elif 'O-Haze' in name:
                net = DM2FNet_woPhy().cuda()
                dataset = OHazeDataset(root, 'test')
            elif 'HazeRD' in name:
                net = DM2FNet().cuda()
                dataset = HazeRDDataset(root, 'data')
            else:
                raise NotImplementedError

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                if 'HazeRD' in name:
                    net.load_state_dict(torch.load(os.path.join(ckpt_path, 'RESIDE_ITS', args['snapshot'] + '.pth')))
                else:
                    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))

            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            psnrs, ssims, mses, ciede2000s = [], [], [], []
            loss_record = AvgMeter()

            # 测试过程计时开始
            test_start_time = time.time()

            for idx, data in enumerate(dataloader):
                haze, gts, fs = data
                haze = haze.cuda()

                # 单个数据处理计时开始
                single_start_time = time.time()

                if 'SOTS' in name:
                    res = net(haze).detach()
                else:
                    res = sliding_forward(net, haze).detach()

                loss = criterion(res, gts.cuda())
                loss_record.update(loss.item(), haze.size(0))

                mse_scores, ciede2000_scores = [], []

                for i in range(len(fs)):
                    r = res[i].cpu().numpy().transpose([1, 2, 0])
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])

                    # Compute MSE
                    mse = mean_squared_error(gt, r)
                    mse_scores.append(mse)

                    # Compute PSNR
                    psnr = peak_signal_noise_ratio(gt, r)
                    psnrs.append(psnr)

                    # Compute SSIM
                    ssim = structural_similarity(gt, r, data_range=1, multichannel=True, win_size=3,
                                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                    ssims.append(ssim)

                    # Compute CIEDE2000
                    ciede2000 = compute_ciede2000_downsampled(r, gt)
                    ciede2000_scores.append(ciede2000)
                    
                    # Print the scores
                    print('predicting for {} ({}/{}) [{}]: MSE {:.4f}, PSNR {:.4f}, SSIM {:.4f}, CIEDE2000 {:.4f}'
                          .format(name, idx + 1, len(dataloader), fs[i], mse, psnr, ssim, ciede2000))

                # 单个数据处理计时结束
                single_elapsed_time = time.time() - single_start_time
                print('Elapsed time for one sample: {:.4f} seconds'.format(single_elapsed_time))
                
                # Calculate mean scores for the dataset
                mean_mse = np.mean(mse_scores)
                mean_ciede2000 = np.mean(ciede2000_scores)
                mses.append(mean_mse)
                ciede2000s.append(mean_ciede2000)

                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(ckpt_path, exp_name,
                                     '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))

            # 测试过程计时结束
            test_elapsed_time = time.time() - test_start_time
            print(f"Testing process took: {test_elapsed_time:.2f} seconds")

            print(f"[{name}] L1: {loss_record.avg:.6f}, MSE: {np.mean(mses):.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, CIEDE2000: {np.mean(ciede2000s):.6f}")

        # 整体过程计时结束
        overall_elapsed_time = time.time() - overall_start_time
        print(f"Overall process took: {overall_elapsed_time:.2f} seconds")
            
if __name__ == '__main__':
    main()