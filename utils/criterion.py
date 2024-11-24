import torch
from torch import nn
from torchvision.models.vgg import vgg16, VGG16_Weights

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

from torchvision import transforms


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        loss_network_1 = nn.Sequential(*list(vgg.features)[:4]).eval()
        for param in loss_network_1.parameters():
            param.requires_grad = False
        self.loss_network_1 = loss_network_1

        loss_network_2 = nn.Sequential(*list(vgg.features)[:9]).eval()
        for param in loss_network_2.parameters():
            param.requires_grad = False
        self.loss_network_2 = loss_network_2

        loss_network_3 = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in loss_network_3.parameters():
            param.requires_grad = False
        self.loss_network_3 = loss_network_3

        loss_network_4 = nn.Sequential(*list(vgg.features)[:23]).eval()
        for param in loss_network_4.parameters():
            param.requires_grad = False
        self.loss_network_4 = loss_network_4


        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

        self.transform = transforms.Compose([
            transforms.Resize(size=(256, 256), antialias=True), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, input, target):

        input = self.transform(input[:, 0:3, :, :])
        target = self.transform(target[:, 0:3, :, :])

        perception_loss = 0.0
        perception_loss += self.l2_loss(self.loss_network_1(input), self.loss_network_1(target))
        perception_loss += self.l2_loss(self.loss_network_2(input), self.loss_network_2(target))
        perception_loss += self.l2_loss(self.loss_network_3(input), self.loss_network_3(target))
        perception_loss += self.l2_loss(self.loss_network_4(input), self.loss_network_4(target))
        
        return perception_loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



if __name__ == '__main__':
    M=PerceptualLoss()
    from torchsummary import summary
    summary(M,[(2,512,512),(3,512,512)],device='cpu')