import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--model_name', default='netG_epoch_4_84.pth', type=str, help='generator model epoch name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    print(MODEL_NAME)

    # results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
    #         'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}
    results = {'psnr': [], 'ssim': []}

    model = Generator(UPSCALE_FACTOR).eval()
    # if torch.cuda.is_available():
    #     model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=torch.device('cpu')))
    # model = torch.load('epochs/' + MODEL_NAME, map_location=torch.device('cpu'))

    test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

    out_path = 'benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        image_name = image_name[0]
        with torch.no_grad():
            lr_image = lr_image
            hr_image = hr_image        
        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()

        sr_image = model(lr_image)
        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

        test_images = torch.stack(
            [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
            display_transform()(sr_image.data.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=3, padding=5)
        new_image_name = image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f.' % (psnr, ssim) + image_name.split('.')[-1]
        utils.save_image(image, out_path + new_image_name , padding=5)

        psnr_value = float(new_image_name.split('_')[2])
        ssim_string = new_image_name.split('_')[4]       
        ssim_value = float(ssim_string.split('.')[0] + '.' + ssim_string.split('.')[1] )

        results['psnr'].append(psnr_value)
        results['ssim'].append(ssim_value)


    out_path = 'statistics/'
    

    data_frame = pd.DataFrame(results, columns=['psnr', 'ssim'])
    data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results.csv', index_label='DataSet')

    df = pd.read_csv("/SRGAN/statistics/srf_4_test_results.csv")
    cols = ["DataSet", "psnr", "ssim"]
    data = df[cols]
    x = data["DataSet"]
    for col in cols[1:]:
        y = data[col]
        plt.figure()
        plt.plot(x, y)
        plt.xlabel("Dataset")
        plt.ylabel(col)
        plt.title(col + " vs Dataset")

        plt.show()


                