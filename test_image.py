import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

# parser = argparse.ArgumentParser(description='Test Single Image')
# parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
# parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
# parser.add_argument('--image_name', type=str, help='test low resolution image name')
# parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
# opt = parser.parse_args()

def generate_image(IMAGE_PATH,filename,MODEL_NAME):

    UPSCALE_FACTOR = 4
    TEST_MODE = False
    IMAGE_NAME = IMAGE_PATH
    MODEL_NAME = MODEL_NAME

    model = Generator(UPSCALE_FACTOR).eval()
    if TEST_MODE:
        model.cuda()
        model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
    else:
        model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))

    image = Image.open(IMAGE_NAME)
    image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()

    # start = time.clock()
    out = model(image)
    # elapsed = (time.clock() - start)
    # print('cost' + str(elapsed) + 's')
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save('static/downloads/'+ 'out_srf_1'+ str(UPSCALE_FACTOR) + '_' + filename)
    out_img_path = 'out_srf_1'+ str(UPSCALE_FACTOR) + '_' + filename

    return out_img_path
