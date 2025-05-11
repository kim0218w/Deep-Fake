import torch
import numpy as np
from PIL import Image
import pickle
import requests
import os

def download_pkl(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading {url} ...")
        r = requests.get(url, stream=True)
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded to {save_path}")
    else:
        print(f"{save_path} already exists.")

def load_generator(network_pkl, device):
    with open(network_pkl, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    G.eval()
    return G

def sample_latent(G, seed=0):
    torch.manual_seed(seed)
    z = torch.randn([1, G.z_dim], device=G.device)
    return z

def generate_image(G, z):
    img = G(z, None, truncation_psi=1.0, noise_mode='const')
    img = (img.clamp(-1, 1) + 1) * 127.5
    img = img[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return img

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pkl_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-1024x1024.pkl"
    pkl_path = "stylegan3-t-ffhqu-1024x1024.pkl"

    # 1. pkl 파일 다운로드
    download_pkl(pkl_url, pkl_path)

    # 2. Generator 로드
    G = load_generator(pkl_path, device)

    # 3. latent 샘플링 및 이미지 생성
    z = sample_latent(G, seed=42)
    img = generate_image(G, z)

    # 4. 이미지 저장
    Image.fromarray(img, 'RGB').save('stylegan3_result.png')
    print("이미지 저장 완료: stylegan3_result.png")
