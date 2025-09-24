# model.py
import torch.nn as nn
import timm
import torch
import os
import streamlit as st # Streamlit을 모델 로딩 시 사용하므로 여기에 포함

class DRModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)

    def forward(self, x):
        return self.model(x)

@st.cache_resource # Streamlit의 캐시 기능을 활용하여 모델을 한 번만 로드
def load_pytorch_models(model_name, model_paths, device):
    models = []
    #st.write(f"{device}에서 XAI 분석을 위한 PyTorch 모델을 로드 중...")
    for path in model_paths:
        model = DRModel(model_name, pretrained=False)
        try:
            if not os.path.exists(path):
                st.warning(f"⚠️ 경고: {path}에서 모델 파일을 찾을 수 없습니다. 해당 모델을 건너뜁니다.")
                continue
            # weights_only=True는 PyTorch 2.0 이상에서 권장되는 방식
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            models.append(model)
            #st.success(f"✅ {path}에서 모델을 성공적으로 로드했습니다.")
        except Exception as e:
            st.error(f"❌ {path}에서 모델 로드 중 오류 발생: {e}")

    if not models:
        st.error("PyTorch 모델을 로드하지 못했습니다. CFG.MODEL_PATHS를 확인하고 모델 파일이 존재하는지 확인하세요.")
        return None
    #st.write(f"{len(models)}개의 PyTorch 모델을 로드했습니다.")
    return models