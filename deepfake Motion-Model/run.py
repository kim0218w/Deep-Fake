import matplotlib
matplotlib.use('Agg') #GUI 없이 이미지 생성을 위한 백엔드 설정  
#matplotlib 설정        
import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
from frames_dataset import FramesDataset
#모델 컴포넌트 임포트
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.bg_motion_predictor import BGMotionPredictor
from modules.dense_motion import DenseMotionNetwork
from modules.avd_network import AVDNetwork
#함수 임포트
import torch
from train import train
from train_avd import train_avd
from reconstruction import reconstruction
import os 


if __name__ == "__main__":
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.9")
#명령줄 인수 파싱 기본적으로 train모드, config/vox-256.yaml설정파일을 사용
    parser = ArgumentParser()
    parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "train_avd"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0,1", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
#설정 파일 로드드
    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
#로그 디렉토리 설정정
    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())
#모델 초기화화
    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
#GPU 설정
    if torch.cuda.is_available():
        cuda_device = torch.device('cuda:'+str(opt.device_ids[0]))
        inpainting.to(cuda_device)

    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])
                                                           
    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])
        dense_motion_network.to(opt.device_ids[0])
#배경 예측기 초기화     
    bg_predictor = None
    if (config['model_params']['common_params']['bg']):
        bg_predictor = BGMotionPredictor()
        if torch.cuda.is_available():
            bg_predictor.to(opt.device_ids[0])
#AVD 네트워크 초기화
    avd_network = None
    if opt.mode == "train_avd":
        avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                             **config['model_params']['avd_network_params'])
        if torch.cuda.is_available():
            avd_network.to(opt.device_ids[0])
#데이터셋 초기화
    dataset = FramesDataset(is_train=(opt.mode.startswith('train')), **config['dataset_params'])
#로그 디렉토리 생성
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)
#모드 선택
    if opt.mode == 'train':
        print("Training...")
        train(config, inpainting, kp_detector, bg_predictor, dense_motion_network, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'train_avd':
        print("Training Animation via Disentaglement...")
        train_avd(config, inpainting, kp_detector, bg_predictor, dense_motion_network, avd_network, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, inpainting, kp_detector, bg_predictor, dense_motion_network, opt.checkpoint, log_dir, dataset)
