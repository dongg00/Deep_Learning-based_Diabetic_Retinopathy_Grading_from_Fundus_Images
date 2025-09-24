# config.py
import torch
import os

class CFG:
    STAGE = 2
    MODEL_NAME = 'efficientnet_b5'
    IMG_SIZE = 456
    BATCH_SIZE = 1
    NUM_WORKERS = 0
    SEED = 42
    MODEL_SAVE_PREFIX = "safety_finetuned"
    N_SPLITS = 3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # CUDA 사용 가능 여부 확인

CFG.MODEL_PATHS = [f"{CFG.MODEL_SAVE_PREFIX}_fold_{i+1}.pth" for i in range(CFG.N_SPLITS)]

# 현재 스크립트가 실행되는 디렉토리 기준으로 모델 경로 설정 (예시)
# script_dir = os.path.dirname(__file__)
# CFG.MODEL_PATHS = [os.path.join(script_dir, "models", f"{CFG.MODEL_SAVE_PREFIX}_fold_{i+1}.pth") for i in range(CFG.N_SPLITS)]

# MedGemma 모델 경로 (필요시 추가)
CFG.MEDGEMMA_MODEL_NAME = "google/medgemma-4b-it"