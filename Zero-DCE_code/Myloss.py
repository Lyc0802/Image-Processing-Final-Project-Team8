import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np


class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)


        return k

			
class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d
        
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)
    def forward(self, x ):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b,c,h,w = x.shape
        # x_de = x.cpu().detach().numpy()
        r,g,b = torch.split(x , 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        # print(k)
        

        k = torch.mean(k)
        return k

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3
    

class L_noise_map_penalty(nn.Module):
    """
    基於 noise map 差異的損失函數
    - 比較 original 和 enhanced 的 noise map
    - 如果 enhanced 的 noise map 變大 → 產生 loss
    - 懲罰噪聲被放大的現象
    """
    def __init__(self, patch_size=8):
        super(L_noise_map_penalty, self).__init__()
        self.patch_size = patch_size
    
    def compute_hybrid_noise_map_torch(self, img):
        """
        PyTorch 版本的 hybrid noise map 計算
        
        Args:
            img: [B, 3, H, W] RGB 圖像，範圍 [0, 1]
        
        Returns:
            noise_map: [B, 1, H, W] 噪聲圖，範圍 [0, 1]
        """
        B, C, H, W = img.shape
        device = img.device
        
        # 轉灰度
        gray = torch.mean(img, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # --- 1. High-pass filter ---
        # 選項 A: 強化版
        hp_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        # 選項 B: 溫和版（可註解/取消註解切換）
        # hp_kernel = torch.tensor([
        #     [0, -1,  0],
        #     [-1, 4, -1],
        #     [0, -1,  0]
        # ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        hp = F.conv2d(gray, hp_kernel, padding=1)
        hp = torch.abs(hp)
        hp = hp / (hp.max() + 1e-6)
        
        # --- 2. Laplacian ---
        lap_kernel = torch.tensor([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        lap = F.conv2d(gray, lap_kernel, padding=1)
        lap = torch.abs(lap)
        lap = lap / (lap.max() + 1e-6)
        
        # --- 3. Patch Variance ---
        patch = self.patch_size
        
        # 計算 padding
        pad_h = (patch - H % patch) % patch
        pad_w = (patch - W % patch) % patch
        
        # Padding (使用 reflect 模式)
        gray_padded = F.pad(gray, (0, pad_w, 0, pad_h), mode='reflect')
        B, _, H_pad, W_pad = gray_padded.shape
        
        # Unfold to patches
        patches = F.unfold(gray_padded, kernel_size=patch, stride=patch)
        # patches shape: [B, patch*patch, num_patches]
        
        # 計算每個 patch 的標準差
        patch_std = torch.std(patches, dim=1)  # [B, num_patches]
        
        # Reshape 回 2D
        num_patches_h = H_pad // patch
        num_patches_w = W_pad // patch
        pv_small = patch_std.view(B, 1, num_patches_h, num_patches_w)
        
        # Resize 回原始尺寸
        pv = F.interpolate(pv_small, size=(H, W), mode='bilinear', align_corners=False)
        pv = pv / (pv.max() + 1e-6)
        
        # --- Fuse ---
        N = 0.25 * hp + 0.25 * lap + 0.50 * pv
        N = N / (N.max() + 1e-6)
        
        # --- Smooth ---
        # 高斯模糊 (7x7, sigma≈1.5)
        gaussian_kernel = self._get_gaussian_kernel(7, 1.5, device)
        N = F.conv2d(N, gaussian_kernel, padding=3)
        
        return N
    
    def _get_gaussian_kernel(self, kernel_size, sigma, device):
        """生成高斯核"""
        x = torch.arange(kernel_size, dtype=torch.float32, device=device)
        x = x - kernel_size // 2
        gauss = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d.view(1, 1, kernel_size, kernel_size)
    
    def forward(self, original, enhanced, original_noise_map=None):
        """
        Args:
            original: 原始圖像 [B, 3, H, W]
            enhanced: 增強後圖像 [B, 3, H, W]
            original_noise_map: (可選) 預計算的原始 noise map [B, 1, H, W]
                               如果提供，則不重新計算
        
        Returns:
            loss: 噪聲放大的懲罰損失
        """
        # 計算或使用預計算的 original noise map
        if original_noise_map is None:
            noise_map_original = self.compute_hybrid_noise_map_torch(original)
        else:
            noise_map_original = original_noise_map
        
        # 計算 enhanced 的 noise map
        noise_map_enhanced = self.compute_hybrid_noise_map_torch(enhanced)
        
        # 計算差異：如果 enhanced 的噪聲更大 → loss
        noise_diff = noise_map_enhanced - noise_map_original
        
        # 只懲罰噪聲增加的部分
        # noise_increase = F.relu(noise_diff)
        noise_increase = torch.pow(F.relu(noise_diff), 2)
        
        # 平均損失
        loss = torch.mean(noise_increase)
        
        return loss
