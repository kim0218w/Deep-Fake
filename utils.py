from torch.utils.data import DataLoader

# 이미지 경로 리스트 (여기선 예시로 사용)
image_paths = ["./data/image1.jpg", "./data/image2.jpg", "./data/image3.jpg"]

# 데이터셋 객체 생성
dataset = CustomDataset(image_paths, transform=transform)

# 데이터로더 객체 생성
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
