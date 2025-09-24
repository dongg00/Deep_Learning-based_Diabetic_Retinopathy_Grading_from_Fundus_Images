# ====================================================
# 1. 라이브러리 임포트 및 기본 설정
# ====================================================
# --- 기본 시스템 및 데이터 처리 라이브러리 ---
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1' # Albumentations 라이브러리의 업데이트 확인 메시지를 비활성화.
import sys, gc, random
from functools import partial
import numpy as np, pandas as pd, cv2

# --- 유틸리티 및 시각화 라이브러리 ---
from tqdm.auto import tqdm # 반복문의 진행 상황을 시각적인 progress bar로 보여줌.
from sklearn.model_selection import StratifiedKFold # 교차 검증 시 클래스 비율을 유지하며 데이터를 분할.
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix # 모델 성능 평가 지표 계산.
import scipy as sp
import matplotlib.pyplot as plt, seaborn as sns

# --- PyTorch 딥러닝 라이브러리 ---
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader # 사용자 정의 데이터셋을 만들고, 데이터를 배치 단위로 효율적으로 불러오기 위한 도구.
import timm # 사전 학습된 다양한 최신 이미지 모델을 쉽게 사용할 수 있는 라이브러리.
from torch.optim import AdamW # 딥러닝 모델의 가중치를 업데이트하는 최적화 알고리즘.
from torch.optim.lr_scheduler import OneCycleLR # 학습 과정 동안 학습률(learning rate)을 동적으로 조절하는 스케줄러.

# --- 데이터 증강 라이브러리 ---
import albumentations as A
from albumentations.pytorch import ToTensorV2 # Albumentations 변환 후 이미지를 PyTorch 텐서로 변환.

def seed_everything(seed):
    """
    실험의 재현성을 보장하기 위해 모든 무작위성을 통제하는 함수.
    이 함수를 호출하면 언제, 어디서 실행하든 동일한 결과를 얻을 수 있어 신뢰도 있는 실험이 가능.
    """
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ====================================================
# 2. 설정 (Configuration) - 2단계 학습용
# ====================================================
# 프로젝트의 모든 하이퍼파라미터와 설정을 한 곳에서 관리하는 '중앙 통제실'.
class CFG:
    # --- 현재 실행할 단계를 선택 ---
    # 1단계: 대규모 EyePACS 데이터셋으로 모델의 기초 실력을 쌓는 '사전 학습' 단계.
    # 2단계: 1단계에서 만든 모델을 가져와, 더 작고 품질 좋은 APTOS 데이터셋으로 정교하게 다듬는 '미세 조정' 단계.
    STAGE = 1
    
    # --- 공통 설정 ---
    MODEL_NAME = 'efficientnet_b5' # 사용할 사전 학습 모델의 이름.
    IMG_SIZE = 456 # 모델에 입력할 이미지의 크기. B5 모델의 표준 입력 크기.
    BATCH_SIZE = 8 # 한 번의 학습에 사용할 이미지의 수. 모델과 이미지가 크므로 GPU 메모리에 맞춰 조정.
    EPOCHS = 15 # 전체 데이터셋을 최대 몇 번 반복하여 학습할지 결정. (조기 종료 기능이 제어)
    NUM_WORKERS = 4 # 데이터를 불러올 때 사용할 CPU 프로세스의 수.
    SEED = 42 # 재현성을 위한 랜덤 시드 값.
    N_SPLITS = 3 # 교차 검증을 위해 데이터를 나눌 폴드(fold)의 수.
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PATIENCE = 3 # 검증 성능이 3 에포크 동안 개선되지 않으면 해당 폴드의 학습을 조기 중단.

    # --- 1단계: EyePACS 사전학습용 설정 ---
    if STAGE == 1:
        TRAIN_CSV = "data/train_cleaned.csv" # EyePACS 데이터 경로.
        LR = 1e-4 # 비교적 높은 학습률로 모델이 빠르게 일반적인 특징을 학습하도록 설정.
        WEIGHT_DECAY = 1e-6
        MODEL_SAVE_PREFIX = "model" # 저장될 모델 파일 이름의 접두사.

    # --- 2단계: APTOS 미세조정용 설정 ---
    elif STAGE == 2:
        TRAIN_CSV = "data/test.csv" # APTOS 데이터 경로.
        LR = 1e-5  # 이미 학습된 지식을 미세하게 조정하기 위해 훨씬 낮은 학습률 사용.
        WEIGHT_DECAY = 1e-6
        MODEL_SAVE_PREFIX = "final_model" # 최종 모델 파일 이름의 접두사.
        # 2단계 학습을 시작할 때 불러올 1단계 모델들의 경로를 지정.
        PRETRAINED_PATHS = [f"model/model_fold_{i}.pth" for i in range(1, N_SPLITS + 1)]

seed_everything(CFG.SEED)

# ====================================================
# 3. 전처리, 데이터셋, 증강, 모델
# ====================================================
def crop_image_to_circle(image):
    """안저 이미지 주변의 불필요한 검은색 배경을 제거하고, 원형 안구 영역만 잘라냄."""
    if image.ndim == 3: gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else: gray_image = image
    blurred = cv2.GaussianBlur(gray_image, (0,0), 10)
    _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return image
    cnt = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center, radius = (int(x), int(y)), int(radius)
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, center, radius, 255, -1)
    x, y, w, h = cv2.boundingRect(cnt)
    cropped_image, mask = image[y:y+h, x:x+w], mask[y:y+h, x:x+w]
    final_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
    return final_image

def ben_graham_preprocessing(image, sigmaX=10):
    """Ben Graham 기법. 이미지의 대비를 극대화하여 미세혈관류, 출혈 등 병변을 두드러지게 함."""
    return cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX), -4, 128)

class DRDataset(Dataset):
    """PyTorch의 DataLoader가 데이터를 어떻게 불러올지 정의하는 '데이터 공급 설명서'."""
    def __init__(self, df, image_path_col, label_col, transforms=None, is_test=False):
        self.df, self.image_paths = df, df[image_path_col].values
        self.transforms, self.is_test = transforms, is_test
        if not self.is_test: self.labels = df[label_col].values

    def __len__(self): return len(self.df) # 데이터셋의 총 길이를 반환.
    def __getitem__(self, idx):
        """특정 인덱스(idx)의 데이터를 요청받았을 때 실행되는 함수."""
        file_path = self.image_paths[idx]
        image = cv2.imread(file_path)
        if image is None: return torch.zeros((3, CFG.IMG_SIZE, CFG.IMG_SIZE)), torch.tensor(0.0, dtype=torch.float)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV(BGR) -> 일반(RGB) 순서로 변환.
        image = crop_image_to_circle(image) # 1차 전처리.
        image = ben_graham_preprocessing(image) # 2차 전처리.
        if self.transforms: image = self.transforms(image=image)['image'] # 데이터 증강 적용.
        if self.is_test: return image # 테스트(OOF 예측) 시에는 이미지만 반환.
        label = torch.tensor(self.labels[idx], dtype=torch.float) # 정답 레이블을 텐서로 변환.
        return image, label

def get_transforms(data_type, img_size):
    """학습/검증 단계에 따라 다른 데이터 증강 파이프라인을 반환하는 함수."""
    if data_type == 'train': # 학습 데이터에는 다양한 변형을 적용하여 모델의 일반화 성능을 높임.
        return A.Compose([
            A.Resize(img_size, img_size), A.RandomRotate90(p=0.5), A.Flip(p=0.5), A.Transpose(p=0.5),
            A.ShiftScaleRotate(p=0.7), A.RandomBrightnessContrast(p=0.7), A.CoarseDropout(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2(),
        ])
    elif data_type == 'valid': # 검증 데이터에는 무작위 변형을 적용하지 않음 (일관된 평가를 위함).
        return A.Compose([ A.Resize(img_size, img_size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2() ])

class DRModel(nn.Module):
    """PyTorch의 nn.Module을 상속받아 우리만의 딥러닝 모델 클래스를 정의."""
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        # timm 라이브러리를 통해 EfficientNet-B5 모델을 생성.
        # pretrained=True: ImageNet으로 미리 학습된 가중치를 가져와서 사용 (전이 학습).
        # num_classes=1: 모델의 최종 출력을 1개로 설정하여 '회귀(Regression)' 문제로 접근.
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
    def forward(self, x): return self.model(x)

class OptimizedRounder(object):
    """회귀 모델의 연속적인 예측값을 QWK 점수를 최대화하는 이산적인 등급으로 변환하는 후처리 클래스."""
    def __init__(self): self.coef_ = 0
    def _kappa_loss(self, coef, X, y):
        X_p = self.predict(X, coef); ll = cohen_kappa_score(y, X_p, weights='quadratic'); return -ll
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y); initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]: X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]: X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]: X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]: X_p[i] = 3
            else: X_p[i] = 4
        return X_p.astype(int)
    def coefficients(self): return self.coef_['x']

# ====================================================
# 4. 학습 및 검증 함수 (TTA 로직 추가)
# ====================================================
def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, class_weights):
    """1 에포크(Epoch) 동안의 학습 과정을 정의하는 함수."""
    model.train(); running_loss = 0.0
    pbar = tqdm(train_loader, desc="Training", file=sys.stdout)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device).view(-1, 1)
        optimizer.zero_grad(); outputs = model(images); loss = criterion(outputs, labels)
        weights = class_weights[labels.squeeze().long()].view(-1, 1) # 각 샘플의 정답에 해당하는 가중치를 가져옴.
        weighted_loss = (loss * weights).mean() # 손실에 가중치를 곱하여 최종 손실 계산.
        weighted_loss.backward(); optimizer.step(); scheduler.step() # 역전파 및 가중치 업데이트.
        running_loss += weighted_loss.item() * images.size(0)
        pbar.set_postfix(loss=weighted_loss.item())
    return running_loss / len(train_loader.dataset)

def validate_one_epoch(model, valid_loader, criterion, device):
    """1 에포크 동안의 검증 과정을 정의하는 함수."""
    model.eval(); running_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad(): # 그래디언트 계산 비활성화.
        pbar = tqdm(valid_loader, desc="Validating", file=sys.stdout)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).view(-1, 1)
            outputs = model(images); loss = criterion(outputs, labels).mean() 
            running_loss += loss.item() * images.size(0)
            all_preds.append(outputs.cpu().numpy()); all_labels.append(labels.cpu().numpy())
            pbar.set_postfix(loss=loss.item())
    epoch_loss = running_loss / len(valid_loader.dataset)
    all_preds = np.concatenate(all_preds).flatten(); all_labels = np.concatenate(all_labels).flatten()
    temp_preds = all_preds.clip(0, 4).round() # 임시 반올림으로 kappa 점수 계산.
    kappa = cohen_kappa_score(all_labels, temp_preds, weights='quadratic')
    return epoch_loss, kappa, all_preds, all_labels

def predict_with_tta(model, data_loader, device):
    """테스트 시점 증강(TTA)을 적용하여 예측의 안정성을 높이는 함수."""
    model.eval(); all_preds = []
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="TTA Prediction", file=sys.stdout)
        for images in pbar:
            images = images.to(device)
            pred1 = model(images) # 원본 이미지 예측.
            images_flipped = torch.flip(images, dims=[3]) # 좌우 반전 이미지 생성.
            pred2 = model(images_flipped) # 좌우 반전 이미지 예측.
            avg_pred = (pred1 + pred2) / 2 # 두 예측의 평균을 최종 예측값으로 사용.
            all_preds.append(avg_pred.cpu().numpy())
    return np.concatenate(all_preds).flatten()

# ====================================================
# 5. 메인 학습 루프 (최종 검수 완료)
# ====================================================
def run_training():
    """전체 학습 과정을 총괄하는 메인 함수."""
    train_df = pd.read_csv(CFG.TRAIN_CSV)
    print(f"--- STAGE {CFG.STAGE} ---")
    print(f"학습 데이터 로드 완료: {CFG.TRAIN_CSV}, 총 {len(train_df)}개")

    # --- 스무딩 가중 손실 계산 ---
    class_counts = train_df['level'].value_counts().sort_index()
    # np.log1p를 사용하여 클래스 간 가중치 차이를 부드럽게 만듦.
    weights = 1.0 / np.log1p(class_counts)
    class_weights = (weights / weights.sum()) * len(class_counts)
    class_weights = torch.tensor(class_weights.values, dtype=torch.float).to(CFG.DEVICE)
    print(f"적용된 스무딩 가중치: {class_weights.cpu().numpy().round(2)}")
    
    skf = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.SEED)
    oof_preds, oof_labels = np.zeros(len(train_df)), np.zeros(len(train_df))

    # --- 모든 폴드의 학습 기록을 저장할 딕셔너리 ---
    history_all_folds = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['level'])):
        print(f"\n========== Fold: {fold+1}/{CFG.N_SPLITS} ==========")
        train_fold_df, valid_fold_df = train_df.iloc[train_idx], train_df.iloc[val_idx]
        
        train_dataset = DRDataset(train_fold_df, 'image_path', 'level', transforms=get_transforms('train', CFG.IMG_SIZE))
        valid_dataset = DRDataset(valid_fold_df, 'image_path', 'level', transforms=get_transforms('valid', CFG.IMG_SIZE), is_test=False)
        tta_dataset = DRDataset(valid_fold_df, 'image_path', 'level', transforms=get_transforms('valid', CFG.IMG_SIZE), is_test=True)
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
        tta_loader = DataLoader(tta_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS, pin_memory=True)
        
        # 1단계는 ImageNet 사전학습 모델로 시작, 2단계는 1단계 모델을 불러와서 시작.
        model = DRModel(CFG.MODEL_NAME, pretrained=(CFG.STAGE == 1)).to(CFG.DEVICE)
        if CFG.STAGE == 2:
            try:
                pretrained_path = CFG.PRETRAINED_PATHS[fold]
                model.load_state_dict(torch.load(pretrained_path, map_location=CFG.DEVICE))
                print(f"  -> Successfully loaded pretrained weights from: {pretrained_path}")
            except FileNotFoundError:
                print(f"  -> WARNING: Pretrained weights not found at {pretrained_path}. Starting from ImageNet weights.")

        criterion = nn.MSELoss(reduction='none') # 각 샘플의 손실을 개별적으로 계산하도록 설정.
        optimizer = AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = OneCycleLR(optimizer, max_lr=CFG.LR, steps_per_epoch=len(train_loader), epochs=CFG.EPOCHS)
        
        best_kappa = 0; patience_counter = 0
        
        # --- 현재 폴드의 학습 기록을 저장할 리스트 ---
        fold_history = {'train_loss': [], 'valid_loss': [], 'kappa': []}

        for epoch in range(CFG.EPOCHS):
            print(f"--- Epoch {epoch+1}/{CFG.EPOCHS} ---")
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, CFG.DEVICE, class_weights)
            valid_loss, kappa, _, _ = validate_one_epoch(model, valid_loader, criterion, CFG.DEVICE)
            print(f"  Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Kappa: {kappa:.4f}")
            
            # --- 현재 에포크의 결과를 history에 저장 ---
            fold_history['train_loss'].append(train_loss)
            fold_history['valid_loss'].append(valid_loss)
            fold_history['kappa'].append(kappa)

            if kappa > best_kappa:
                best_kappa = kappa
                # --- 모델 저장 경로 확인 및 생성 ---
                save_path = f"model/{CFG.MODEL_SAVE_PREFIX}_fold_{fold+1}.pth"
                os.makedirs(os.path.dirname(save_path), exist_ok=True) # 'model' 폴더가 없으면 자동 생성.
                torch.save(model.state_dict(), save_path)
                print(f"  -> Best model saved! Kappa: {best_kappa:.4f}"); patience_counter = 0
            else:
                patience_counter += 1; print(f"  -> No improvement. Patience: {patience_counter}/{CFG.PATIENCE}")
            
            # --- 조기 종료 로직 ---
            if patience_counter >= CFG.PATIENCE:
                print(f"  !!! Early stopping triggered after {epoch+1} epochs. !!!"); break
        
        # --- 현재 폴드의 최종 history를 전체 history에 저장 ---
        history_all_folds[f"Fold {fold+1}"] = fold_history

        print(f"Loading best model for fold {fold+1} to generate OOF predictions with TTA...")
        best_model_path = f"model/{CFG.MODEL_SAVE_PREFIX}_fold_{fold+1}.pth"
        model.load_state_dict(torch.load(best_model_path, map_location=CFG.DEVICE))
        
        # --- OOF 예측 생성 시 TTA 적용 ---
        fold_preds = predict_with_tta(model, tta_loader, CFG.DEVICE)
        
        oof_preds[val_idx] = fold_preds
        oof_labels[val_idx] = valid_fold_df['level'].values
        del model, train_loader, valid_loader; gc.collect(); torch.cuda.empty_cache()

    # ====================================================
    # 6. 후처리 및 최종 평가
    # ====================================================
    print("\n========== Threshold Optimization & Final Score ==========")
    optR = OptimizedRounder(); optR.fit(oof_preds, oof_labels)
    coefficients = optR.coefficients(); print(f"Optimized Coefficients: {coefficients}")
    final_oof_preds = optR.predict(oof_preds, coefficients)
    final_kappa = cohen_kappa_score(oof_labels, final_oof_preds, weights='quadratic')
    print(f"\nFinal OOF Kappa Score after optimization: {final_kappa:.4f}")

    print("\n========== Detailed Performance Analysis ==========")
    class_names = ['0: Normal', '1: Mild', '2: Moderate', '3: Severe', '4: Proliferative']
    report = classification_report(oof_labels, final_oof_preds, target_names=class_names, zero_division=0)
    print("\n[Classification Report]"); print(report)
    print("\n[Confusion Matrix]")
    cm = confusion_matrix(oof_labels, final_oof_preds)
    plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (OOF Predictions)'); plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.show()

    # ====================================================
    # 7. 학습 과정 시각화
    # ====================================================
    print("\n========== Training History Visualization ==========")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Training History - Stage {CFG.STAGE}', fontsize=16)

    for fold_name, history in history_all_folds.items():
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'o--', label=f'{fold_name} Train Loss')
        ax1.plot(epochs, history['valid_loss'], 'o-', label=f'{fold_name} Valid Loss')
        ax2.plot(epochs, history['kappa'], 'o-', label=f'{fold_name} Kappa')

    ax1.set_title('Train & Validation Loss'); ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True)
    ax2.set_title('Validation Kappa Score'); ax2.set_xlabel('Epochs'); ax2.set_ylabel('Kappa Score')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    run_training()