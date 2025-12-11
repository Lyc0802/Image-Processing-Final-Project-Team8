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
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import cv2


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
        # hp = np.abs(hp)
        hp /= (hp.max() + 1e-6)

        # --- 2. Laplacian ---
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        # lap = np.abs(lap)
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
        #N = 0.25 * hp + 0.25 * lap + 0.50 * pv
        N = lap * pv
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
 
def lowlight(image_path, model_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	
	img = cv2.imread(image_path)
	N = compute_hybrid_noise_map(img)    # (H,W)
	A = compute_attenuation_map(N)          # (H,W)
	A_tensor = torch.from_numpy(A).float().to("cuda")   # (H, W)
	A_tensor = A_tensor.unsqueeze(0).unsqueeze(0)        # (1, 1, H, W)

	data_lowlight = Image.open(image_path)

	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	DCE_net = model.enhance_net_nopool().cuda()
	DCE_net.load_state_dict(torch.load(model_path)) 
	start = time.time()
	_,enhanced_image,_ = DCE_net(data_lowlight)

	end_time = (time.time() - start)
	print(end_time)
	image_path = image_path.replace('test_data','result')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
# test_images
	with torch.no_grad():
		filePath = 'data/test_data/'
		modelPath = 'snapshots/Epoch99.pth'
	
		file_list = os.listdir(filePath)

		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*") 
			for image in test_list:
				# image = image
				print(image)
				lowlight(image, modelPath)

		

