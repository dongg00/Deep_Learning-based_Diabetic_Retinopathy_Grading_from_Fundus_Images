# xai_analysis.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image


def analyze_gradcam_heatmap(heatmap, img_shape):
    """
    GradCAM 히트맵을 분석하여 주요 활성화 영역과 색상 강도를 텍스트로 변환
    """
    if heatmap is None or np.max(heatmap) == 0:
        return "히트맵 활성화 영역이 감지되지 않았습니다."
    
    heatmap_normalized = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
    threshold = 0.7 
    
    height, width = img_shape[:2]
    regions = {
        "우측 상단": (0, height//2, width//2, width),
        "좌측 상단": (0, height//2, 0, width//2),
        "우측 하단": (height//2, height, width//2, width),
        "좌측 하단": (height//2, height, 0, width//2),
        "중앙": (height//4, 3*height//4, width//4, 3*width//4)
    }
    
    activation_info = []
    for region_name, (y1, y2, x1, x2) in regions.items():
        region_heatmap = heatmap_normalized[y1:y2, x1:x2]
        if np.any(region_heatmap > threshold):
            mean_intensity = np.mean(region_heatmap[region_heatmap > threshold])
            if mean_intensity > 0.9:
                color_desc = "진한 붉은색"
            elif mean_intensity > 0.8:
                color_desc = "밝은 붉은색"
            elif mean_intensity > 0.7:
                color_desc = "노란색"
            else:
                color_desc = "녹색-청록색"
            activation_info.append(
                f"- [히트맵 활성화 위치]: {region_name}\n"
                f"  - [히트맵 색상]: {color_desc}\n"
                f"  - [원본 이미지 병변]: {region_name}에서 활성화된 영역은 미세출혈, 삼출물 또는 신생혈관으로 의심됩니다."
            )
    
    if not activation_info:
        return "히트맵에서 유의미한 활성화 영역이 감지되지 않았습니다."
    return "\n".join(activation_info)

def make_gradcam_heatmap_pytorch_ensemble(img_tensor_normalized, models, target_layer_name):
    all_heatmaps = []
    for model in models:
        target_layers = []
        # EfficientNet 모델의 특정 레이어를 찾아 Grad-CAM에 사용
        if hasattr(model.model, 'conv_head') and target_layer_name == 'conv_head':
            target_layers = [model.model.conv_head]
        elif hasattr(model.model, 'blocks') and target_layer_name == 'blocks_last_block':
            target_layers = [model.model.blocks[-1]]
        else:
            # 기본적으로 마지막 Conv2d 레이어를 찾으려는 시도
            last_module = None
            for name, module in model.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)) and not isinstance(module, nn.Identity) and len(list(module.children())) == 0:
                    last_module = module
            if last_module:
                target_layers = [last_module]
            else:
                # Fallback: 모델의 마지막 블록을 기본 타겟으로 설정 (timm 모델의 일반적인 구조)
                if hasattr(model.model, 'feature_info'): # timm 모델의 경우
                    try:
                        # 'blocks' 속성이 없으면 'layers' 또는 다른 구조를 따를 수 있음
                        # 여기서는 EfficientNet 계열의 'blocks'를 가정
                        if hasattr(model.model, 'blocks') and len(model.model.blocks) > 0:
                            target_layers = [model.model.blocks[-1]]
                        else: # 다른 모델 구조를 위한 일반적인 마지막 레이어 추론
                            st.warning(f"Grad-CAM을 위한 적절한 마지막 피처 레이어를 동적으로 찾을 수 없습니다. 모델 '{model.model.__class__.__name__}'에 대해 다른 레이어 탐색 로직이 필요할 수 있습니다.")
                            continue
                    except Exception as e:
                        st.warning(f"Grad-CAM 타겟 레이어 설정 중 오류 발생: {e}. 해당 모델을 건너뜁니다.")
                        continue
                else:
                    st.warning(f"Grad-CAM을 위한 적절한 마지막 피처 레이어를 동적으로 찾을 수 없습니다. 해당 모델을 건너뜁니다.")
                    continue
        
        if not target_layers:
            st.warning(f"{target_layer_name}에 대해 유효한 타겟 레이어가 없어 Grad-CAM을 건너뜁니다.")
            continue
            
        cam = GradCAM(model=model, target_layers=target_layers)
        # targets=None은 가장 높은 스코어를 가진 클래스에 대한 CAM을 계산
        grayscale_cam = cam(input_tensor=img_tensor_normalized, targets=None)
        all_heatmaps.append(grayscale_cam[0])

    if not all_heatmaps:
        return np.zeros(img_tensor_normalized.shape[2:]) # 모든 히트맵 생성이 실패한 경우

    avg_heatmap = np.mean(all_heatmaps, axis=0)
    avg_heatmap = np.maximum(avg_heatmap, 0) # 음수 값 제거
    if np.max(avg_heatmap) > 0:
        avg_heatmap = avg_heatmap / np.max(avg_heatmap) # 정규화
    return avg_heatmap

def display_xai_results(original_img_0_1_rgb, gradcam_heatmap, predicted_label_name, alpha=0.4):
    fig, axes = plt.subplots(1, 2, figsize=(4, 4))
    axes[0].imshow(original_img_0_1_rgb)
    axes[0].set_title("병변 탐지 및 분석을 위해 보정된 안저 이미지", fontsize=4)
    axes[0].axis('off')

    if gradcam_heatmap is not None and np.max(gradcam_heatmap) > 0:
        heatmap_resized = cv2.resize(gradcam_heatmap, (original_img_0_1_rgb.shape[1], original_img_0_1_rgb.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        original_img_uint8_bgr = cv2.cvtColor(np.uint8(original_img_0_1_rgb * 255), cv2.COLOR_RGB2BGR)
        superimposed_img_bgr = cv2.addWeighted(original_img_uint8_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)
        axes[1].imshow(superimposed_img_rgb)
        axes[1].set_title(f"Sweet Vision의 분석 핵심 영역", fontsize=4)
        axes[1].axis('off')
    else:
        axes[1].set_title("Grad-CAM (생성되지 않음)")
        axes[1].axis('off')
    plt.tight_layout()
    return fig

def create_gradcam_overlay_image_for_llm(background_image_rgb, heatmap, alpha=0.4):
    if heatmap is None or np.max(heatmap) == 0:
        return None
    
    # 배경 이미지가 0-1 스케일일 경우 0-255로 변환
    if background_image_rgb.dtype == np.float32 and np.max(background_image_rgb) <= 1.0:
        background_image_rgb_uint8 = np.uint8(background_image_rgb * 255)
    else:
        background_image_rgb_uint8 = np.uint8(background_image_rgb)

    heatmap_resized = cv2.resize(heatmap, (background_image_rgb_uint8.shape[1], background_image_rgb_uint8.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    original_img_uint8_bgr = cv2.cvtColor(background_image_rgb_uint8, cv2.COLOR_RGB2BGR)
    superimposed_img_bgr = cv2.addWeighted(original_img_uint8_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(superimposed_img_rgb)