import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np
from torchvision import transforms
import cv2
from tqdm import tqdm

def compute_hybrid_noise_map(img, patch=8):
    """
    給一張 BGR 彩色圖
    對每一個 channel 分別生成 noise map
    回傳 3-channel noise map (float32, 0~1)
    """

    # 拆成 B, G, R 三張灰階影像
    channels = cv2.split(img)

    noise_maps = []

    for ch in channels:
        gray = ch.astype(np.float32)

        # --- 1. High-pass ---
        hp_kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)

        hp = cv2.filter2D(gray, -1, hp_kernel)
        hp = np.abs(hp)
        hp /= (hp.max() + 1e-6)

        # --- 2. Laplacian ---
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap = np.abs(lap)
        lap /= (lap.max() + 1e-6)

        # --- 3. Patch Variance ---
        H, W = gray.shape
        pad_h = (patch - H % patch) % patch
        pad_w = (patch - W % patch) % patch

        gray_padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode='reflect')
        H_pad, W_pad = gray_padded.shape

        pv_small = np.zeros((H_pad//patch, W_pad//patch))
        for i in range(0, H_pad, patch):
            for j in range(0, W_pad, patch):
                p = gray_padded[i:i+patch, j:j+patch]
                pv_small[i//patch, j//patch] = p.std()

        pv = cv2.resize(pv_small / (pv_small.max() + 1e-6), (W, H))

        # --- Fuse channels ---
        N = 0.25 * hp + 0.25 * lap + 0.50 * pv
        N /= (N.max() + 1e-6)

        # --- Smooth ---
        N = cv2.GaussianBlur(N, (7, 7), 0)

        noise_maps.append(N)

    # 合併成 3-channel noise map
    N_color = cv2.merge(noise_maps)

    return N_color

# def compute_hybrid_noise_map(img, patch=8):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
#     # --- 1. High-pass (調整參數選項) ---
#     # 選項 A: 原始強化版
#     # hp_kernel = np.array([
#     #     [-1, -1, -1],
#     #     [-1,  8, -1],
#     #     [-1, -1, -1]
#     # ], dtype=np.float32)
    
#     # 選項 B: 溫和版（建議用於一般情況）
#     hp_kernel = np.array([
#         [0, -1,  0],
#         [-1, 4, -1],
#         [0, -1,  0]
#     ], dtype=np.float32)
    
#     hp = cv2.filter2D(gray, -1, hp_kernel)
#     hp = np.abs(hp)
#     hp /= (hp.max() + 1e-6)
    
#     # --- 2. Laplacian ---
#     lap = cv2.Laplacian(gray, cv2.CV_32F)
#     lap = np.abs(lap)
#     lap /= (lap.max() + 1e-6)
    
#     # --- 3. Patch Variance (修正邊界) ---
#     H, W = gray.shape
    
#     # 計算需要 padding 的大小
#     pad_h = (patch - H % patch) % patch
#     pad_w = (patch - W % patch) % patch
    
#     # Padding（使用 reflect 模式避免邊界偽影）
#     gray_padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode='reflect')
#     H_pad, W_pad = gray_padded.shape
    
#     pv_small = np.zeros((H_pad//patch, W_pad//patch))
#     for i in range(0, H_pad, patch):
#         for j in range(0, W_pad, patch):
#             p = gray_padded[i:i+patch, j:j+patch]
#             pv_small[i//patch, j//patch] = p.std()
    
#     # Resize 回原始尺寸
#     pv = cv2.resize(pv_small / (pv_small.max() + 1e-6), (W, H))
    
#     # --- Fuse them ---
#     N = 0.25 * hp + 0.25 * lap + 0.50 * pv
#     N /= (N.max() + 1e-6)
    
#     # --- Smooth ---
#     N = cv2.GaussianBlur(N, (7, 7), 0)  

#     return N

def compute_attenuation_map(N, k=0.5, min_val=0.3):
	"""
	N: noise map (H×W), 值域應該在 0~1
	k: 控制噪點削弱強度，建議 0.3~0.6
	min_val: A(x) 最低值（避免畫面變太暗）
	"""
	A = 1.0 - k * N
	A = np.clip(A, min_val, 1.0)

	return A


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):

	os.environ['CUDA_VISIBLE_DEVICES']='0'

	DCE_net = model.enhance_net_nopool().cuda()

	DCE_net.apply(weights_init)
	if config.load_pretrain == True:
		DCE_net.load_state_dict(torch.load(config.pretrain_dir))
	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)		
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)


	L_color = Myloss.L_color()
	L_spa = Myloss.L_spa()

	L_exp = Myloss.L_exp(16,0.6)
	L_TV = Myloss.L_TV()
	L_noise_map = Myloss.L_noise_map_penalty(patch_size=8)

	optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	DCE_net.train()

	for epoch in tqdm(range(config.num_epochs)):
		for iteration, img_lowlight in enumerate(train_loader):
			
			img_numpy = img_lowlight.detach().cpu().numpy()   # (B,3,H,W)
			A_list = []

			for b in range(img_numpy.shape[0]):
				img_np = img_numpy[b]               # (3,H,W)
				img_np = np.transpose(img_np, (1,2,0))  # (H,W,3)
				img_np = (img_np * 255).astype(np.uint8)

				N = compute_hybrid_noise_map(img_np)    # (H,W)
				A = compute_attenuation_map(N)          # (H,W)
				A_list.append(A)

			# Stack all A into (B,H,W)
			A_np = np.stack(A_list, axis=0)            # (B,H,W)

			# Convert to torch and reshape to (B,1,H,W)
			A_tensor = torch.from_numpy(A_np).float().to("cuda")   # (B,H,W)
			A_tensor = A_tensor.unsqueeze(1)            # (B,1,H,W)

			img_lowlight = img_lowlight.cuda()

			enhanced_image_1,enhanced_image,A  = DCE_net(img_lowlight)

			Loss_TV = 200*L_TV(A)
			
			loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))

			loss_col = 5*torch.mean(L_color(enhanced_image))
   
   			# loss 1: use  "noise_increase = F.relu(noise_diff)" in L_noise_map_penalty

			# loss_exp = 19*torch.mean(L_exp(enhanced_image))

			# loss_noise_map = 30 * (torch.mean(L_noise_map(img_lowlight, enhanced_image)))
   
			# loss 2: use  "noise_increase = torch.pow(F.relu(noise_diff), 2)" in L_noise_map_penalty

			loss_exp = 10*torch.mean(L_exp(enhanced_image))

			loss_noise_map = 55 * (torch.mean(L_noise_map(img_lowlight, enhanced_image)))
		
			# best_loss
			loss =  Loss_TV + loss_spa + loss_col + loss_exp + loss_noise_map
			
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(DCE_net.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("Loss at iteration", iteration+1, ":", loss.item())
			if ((iteration+1) % config.snapshot_iter) == 0:
				
				torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 		




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch99.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)








	
