# train.py
import torch
from torch.optim import Adam
from models.synthesis_network import SynthesisNetwork
from models.discriminator import Discriminator
from dataset import CustomDataset
from torch.utils.data import DataLoader
from losses.wgan_gp_loss import d_logistic_loss, g_nonsaturating_loss, r1_penalty

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = SynthesisNetwork().to(device)
D = Discriminator().to(device)

g_optim = Adam(G.parameters(), lr=2e-3, betas=(0.0, 0.99))
d_optim = Adam(D.parameters(), lr=2e-3, betas=(0.0, 0.99))

# 데이터셋 로딩
image_paths = ["./data/image1.jpg", "./data/image2.jpg", "./data/image3.jpg"]  # 실제 이미지 경로로 바꿔야 함
dataset = CustomDataset(image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 학습 루프
for epoch in range(epochs):
    for i, real_img in enumerate(dataloader):
        # 실시간 학습을 위해 device로 데이터를 이동
        real_img = real_img.to(device)

        # 랜덤 스타일 벡터 (여기선 간단히 임의로 생성)
        style_vector = torch.randn(real_img.size(0), 512).to(device)

        # 학습 진행 (1단계: Discriminator 학습, 2단계: Generator 학습)
        d_loss, g_loss, r1_loss = train_one_step(G, D, g_optim, d_optim, real_img, style_vector, device)

        # 학습 과정 출력
        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}, R1 Loss: {r1_loss:.4f}")
