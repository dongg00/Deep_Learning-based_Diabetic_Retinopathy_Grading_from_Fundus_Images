import streamlit as st
import numpy as np
import cv2
from PIL import Image
import gc
import torch
import os

# 각 모듈에서 필요한 함수와 클래스 임포트
# (이 부분은 사용자 환경에 따라 실제 파일 경로에 맞게 수정해야 합니다.)
# 이 경로는 사용자 환경에 맞게 정확히 지정되어야 합니다.
from config import CFG
from utils import seed_everything, OptimizedRounder
from preprocessing import crop_image_to_circle, ben_graham_preprocessing, get_transforms
from model import load_pytorch_models, DRModel
# xai_analysis.py는 문제 없다고 하셨으므로 이 임포트와 함수들은 그대로 사용합니다.
from xai_analysis import analyze_gradcam_heatmap, make_gradcam_heatmap_pytorch_ensemble, display_xai_results, create_gradcam_overlay_image_for_llm
from llm_integration import load_medgemma_pipeline, generate_medgemma_report

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_paths = [
    'C:/Windows/Fonts/malgun.ttf', # 윈도우 기본 맑은 고딕
    # 추가적으로 웹폰트나 프로젝트 내부에 폰트 파일을 포함시키는 경우 경로 지정
    # './fonts/NanumGothic.ttf' # 예시: 프로젝트 내 폰트 파일
]

# 사용 가능한 폰트 찾기 및 설정
found_font_path = None
for path in font_paths:
    if os.path.exists(path):
        found_font_path = path
        break

if found_font_path:
    fm.fontManager.addfont(found_font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=found_font_path).get_name()
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
else:
    # 폰트를 찾지 못했을 때 경고 메시지 (선택 사항)
    # st.warning("한글 폰트를 찾을 수 없습니다. 기본 폰트로 표시됩니다.")
    pass

# --- 초기 설정 ---
seed_everything(CFG.SEED)
device = CFG.DEVICE

# Streamlit 세션 상태 초기화 (모델 로드는 앱 시작 시 한 번만 시도)
if 'medgemma_pipe' not in st.session_state:
    #st.info("Sweet Vision의 기능을 준비 중입니다... (최초 실행 시 다소 시간이 걸릴 수 있습니다.)")
    try:
        st.session_state.medgemma_pipe = load_medgemma_pipeline(CFG.MEDGEMMA_MODEL_NAME, device)
        #st.success("MedGemma 모델 로드 완료.")
    except Exception as e:
        st.error(f"MedGemma 모델 로드 중 오류 발생: {e}. AI 소견 기능을 사용할 수 없습니다.")
        st.session_state.medgemma_pipe = None

if 'pytorch_models' not in st.session_state:
    try:
        st.session_state.pytorch_models = load_pytorch_models(CFG.MODEL_NAME, CFG.MODEL_PATHS, device)
        #if st.session_state.pytorch_models is None:
            #st.error("핵심 분석 모델 로드에 실패했습니다. 관리자에게 문의하세요.")
        #else:
            #st.success("PyTorch 모델 로드 완료.")
    except Exception as e:
        st.error(f"PyTorch 모델 로드 중 오류 발생: {e}. 등급 분류 및 시각화 기능을 사용할 수 없습니다.")
        st.session_state.pytorch_models = None

# --- 데이터 변환 정의 ---
pytorch_val_transforms = get_transforms('valid', CFG.IMG_SIZE)

# ====================================================
# Streamlit Application Layout
# ====================================================
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# --- 당뇨망막병증 등급 매핑 딕셔너리 추가 ---
# 이 딕셔너리는 Streamlit 앱의 전역 공간에 정의되어야 합니다.
DR_GRADE_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}
# -----------------------------------------------

# --- CSS 스타일 정의 (탭, 헤더, 사이드바 너비, 파일 업로더 크기 조절) ---
st.markdown("""
<style>
/* 탭 글씨 크기 */
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.5em; /* 탭 글씨 크기 (원하는 값으로 조절) */
    font-weight: bold;
}

/* st.header (h2 태그) 글씨 크기 */
h2 {
    font-size: 3em; /* st.header에 해당하는 h2 태그 글씨 크기 (원하는 값으로 조절) */
    font-weight: bold;
    color: #2E8B57; /* 선택 사항: 색상 변경 */
}

/* st.title (h1 태그) 글씨 크기 (만약 st.title을 사용한다면) */
h1 {
    font-size: 3em; /* h1 태그 글씨 크기 (원하는 값으로 조절) */
    font-weight: bold;
    color: #4682B4; /* 선택 사항: 색상 변경 */
}

/* 사이드바 너비 조절 */
section[data-testid="stSidebar"] {
    width: 700px !important; /* 원하는 너비로 조절 (예: 300px, 400px) */
}

/* 사이드바 파일 업로더 드롭 영역의 최소 높이 조절 (간접적으로 크기 키우기) */
.stFileUploader > div > div:first-child {
    min-height: 180px; /* 원하는 높이로 조절 (기본값보다 크게) */
    border: 2px dashed #4CAF50; /* 경계선 추가 */
    background-color: #e6ffe6; /* 연한 배경색 추가 */
}
.stFileUploader > div > div:first-child p {
    font-size: 1.2em; /* 파일 업로더 텍스트 크기 조절 */
    font-weight: bold;
    color: #4CAF50; /* 파일 업로더 텍스트 색상 조절 */
}
.stFileUploader > div > div:first-child .st-cq { /* Drag and drop text color */
    color: #4CAF50; /* Greenish color for better visibility */
}
/* Browse Files 버튼 텍스트 크기 */
.stFileUploader button p {
    font-size: 1.1em;
}
</style>
""", unsafe_allow_html=True)

# 사이드바에 제목 배치
with st.sidebar:
    # 이미지 파일 경로를 지정합니다.
    # .py 파일과 같은 경로에 있으므로 파일 이름만 명시합니다.
    logo_image_path = "image__24_-removebg-preview.png"
    
    if os.path.exists(logo_image_path):
        # st.image 함수를 사용하여 이미지를 표시합니다.
        # use_column_width=True를 사용하면 사이드바 너비에 맞춰 이미지가 조정됩니다.
        # caption을 추가하여 이미지에 대한 설명을 제공할 수도 있습니다.
        st.image(logo_image_path, use_container_width=True)
    else:
        # 이미지를 찾을 수 없을 경우 기존 텍스트 제목과 경고 메시지를 표시
        st.markdown("<h1 style='font-size: 5em; text-align: center; color: red;'>👁️🗨️Sweet Vision</h1>", unsafe_allow_html=True)
        st.warning(f"로고 이미지를 찾을 수 없습니다: {logo_image_path}")

    st.markdown("---")
    # 파일 업로더 위젯 바로 위에 안내 메시지 추가
    st.markdown("<p style='font-size: 1.2em; font-weight: bold; text-align: center;'>여기에 안저 이미지를 업로드하세요!</p>", unsafe_allow_html=True)

    # 이미지 업로드 위젯은 그대로 사이드바에
    uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])

# 세션 상태에 분석 결과 저장 변수들 초기화 (파일이 새로 업로드되거나 앱이 재시작될 때 초기화)
# 'predicted_label_name'이 다른 탭에서도 유지되어야 하므로, 초기화 로직을 주의깊게 관리
if 'raw_image_rgb_0_255' not in st.session_state:
    st.session_state.raw_image_rgb_0_255 = None
    st.session_state.image_filename = "선택된 파일 없음"
    st.session_state.predicted_label_name = None
    st.session_state.avg_regression_pred = None
    st.session_state.original_img_0_1_rgb = None
    st.session_state.cam_heatmap = None
    st.session_state.medgemma_report_text = None # LLM 결과도 세션 상태에 저장

# 이미지 업로드 처리 로직
if uploaded_file is not None:
    # 새로운 파일이 업로드되었을 때만 세션 상태를 업데이트
    if st.session_state.image_filename != uploaded_file.name:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        try:
            raw_image_bgr_temp = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if raw_image_bgr_temp is None:
                st.error("이미지를 디코딩할 수 없습니다. 유효한 이미지 파일을 업로드하세요.")
                st.session_state.raw_image_rgb_0_255 = None # 오류 시 이미지 상태 초기화
                st.session_state.image_filename = "디코딩 실패"
                # Stop processing for this run to avoid further errors
                st.experimental_rerun() # 앱을 재실행하여 깨끗한 상태로 만듭니다.
            else:
                st.session_state.raw_image_rgb_0_255 = cv2.cvtColor(raw_image_bgr_temp, cv2.COLOR_BGR2RGB)
                st.session_state.image_filename = uploaded_file.name
                # 새 이미지 업로드 시 기존 분석 결과 초기화
                st.session_state.predicted_label_name = None
                st.session_state.avg_regression_pred = None
                st.session_state.original_img_0_1_rgb = None
                st.session_state.cam_heatmap = None
                st.session_state.medgemma_report_text = None # LLM 결과도 초기화
                st.rerun() # 이미지 업로드 시 앱 전체를 다시 실행하여 초기화된 상태로 시작

        except Exception as e:
            st.error(f"이미지 읽기 오류: {e}. 유효한 이미지 파일인지 확인하세요.")
            st.session_state.raw_image_rgb_0_255 = None # 오류 시 이미지 상태 초기화
            st.session_state.image_filename = "읽기 오류"
            st.experimental_rerun() # 앱을 재실행하여 깨끗한 상태로 만듭니다.

# 사이드바에 이미지 표시 (업로드된 파일이 있을 경우)
if st.session_state.raw_image_rgb_0_255 is not None:
    st.sidebar.image(st.session_state.raw_image_rgb_0_255, caption=f"업로드된 이미지: {st.session_state.image_filename}", width=300)
else:
    st.sidebar.info("현재 입력된 이미지가 없습니다.")

# 탭 생성 (이제 3개의 탭만 존재)
tab0, tab1, tab2 = st.tabs(["🏡 홈", "📊 등급 분류", "🔍 판단 근거 시각화 및 소견"])

# --- 탭 0: 홈 탭 ---
with tab0:
    # st.header("🏡 Sweet Vision 소개 및 사용 안내") # st.header 대신 markdown으로 직접 H2 태그 사용
    st.markdown("<h2 style='font-size: 3.5em; font-weight: bold;'>🏡 Sweet Vision 소개 및 사용 안내</h2><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:25px;'>
    <b>Sweet Vision</b>은 인공지능이 안저 이미지를 분석하여 <b>당뇨망막병증 단계를 등급별로 분류</b>하고,<br>
    <b>AI가 왜 그렇게 판단했는지 시각적인 근거</b>를 함께 제시하여 의료진의 <b>신속하고 정확한 진단</b>을 돕는 시스템입니다.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:25px;'>
    <br><b>사용 방법:</b>
    <ol>
        <li>왼쪽 사이드바에서 <b>'Browse files'</b> 버튼을 클릭하여 안저 이미지를 업로드합니다.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.raw_image_rgb_0_255 is None:
        st.info("왼쪽 사이드 바에서 안저 이미지를 업로드하여 분석을 시작하세요.", icon="⬅️")
        st.markdown("---")

    st.markdown("""
    <div style='font-size:25px;'>
    <ol start="2"> <li>이미지 업로드 후, 메인 화면의 <b>각 탭</b>을 클릭하고 <b>해당 기능의 버튼</b>을 눌러 분석 결과를 확인합니다.</li>
        <ul>
            <li><b>🏡 홈:</b> Sweet Vision에 대한 소개와 현재 업로드된 이미지를 확인할 수 있습니다.</li>
            <li><b>📊 등급 분류:</b> 당뇨망막병증의 예측 등급을 확인합니다.</li>
            <li><b>🔍 판단 근거 시각화 및 소견:</b> Sweet Vision이 예측 등급을 판단한 시각적 근거(히트맵)를 확인하고, 생성한 전문적인 당뇨망막병증 소견을 확인합니다.</li>
        </ul>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.raw_image_rgb_0_255 is not None:
        st.markdown(f"---")
        st.markdown(f"**업로드된 이미지:** <span style='font-size:20px;'>`{st.session_state.image_filename}`</span>", unsafe_allow_html=True)
        st.image(st.session_state.raw_image_rgb_0_255, caption="현재 분석 대상 이미지", width=500)

# --- 탭 1: 당뇨망막병증 등급 분류 ---
with tab1:
    st.markdown("<h2 style='font-size: 3.5em; font-weight: bold;'>📊 당뇨망막병증 등급 분류</h2><br>", unsafe_allow_html=True)
    if st.session_state.raw_image_rgb_0_255 is None:
        st.warning("이미지를 업로드하면 당뇨망막병증 등급을 분류할 수 있습니다.")
    elif st.session_state.pytorch_models is None:
        st.error("분류 모델이 로드되지 않아 등급 분류를 수행할 수 없습니다.")
    else:
        if st.button("등급 분류 실행", key="classify_button"):
            with st.spinner("Sweet Vision이 당뇨망막병증 등급을 예측 중입니다..."):
                image_for_model_pred = st.session_state.raw_image_rgb_0_255
                image_for_model_pred = crop_image_to_circle(image_for_model_pred)
                image_for_model_pred = ben_graham_preprocessing(image_for_model_pred)
                
                transformed = pytorch_val_transforms(image=image_for_model_pred)
                img_tensor_normalized = transformed['image'].unsqueeze(0).to(device)

                fold_regression_preds = []
                with torch.no_grad():
                    for model in st.session_state.pytorch_models:
                        output = model(img_tensor_normalized).cpu().numpy().item()
                        fold_regression_preds.append(output)

                st.session_state.avg_regression_pred = np.mean(fold_regression_preds)

                temp_opt_rounder_for_predict = OptimizedRounder()
                temp_opt_rounder_for_predict.coef_ = np.array([0.5, 1.5, 2.5, 3.5])
                predicted_label_idx = temp_opt_rounder_for_predict.predict(np.array([st.session_state.avg_regression_pred]), temp_opt_rounder_for_predict.coef_)[0]
                st.session_state.predicted_label_name = str(predicted_label_idx)
            st.success("등급 예측 완료.")
            # 등급 분류가 완료되면 터미널에 상태를 출력
            print(f"DEBUG main_app (tab1): Predicted label after classification: {st.session_state.predicted_label_name}")

        if st.session_state.predicted_label_name is not None:
            predicted_grade_num = int(st.session_state.predicted_label_name)
            predicted_grade_english = DR_GRADE_NAMES.get(predicted_grade_num, "알 수 없음") # 딕셔너리에 없으면 "알 수 없음"
            st.markdown(f"<p style='font-size:25px;'><b>예측 등급:</b> <span style='color: red;'><b>{predicted_grade_num} 등급 ({predicted_grade_english})</b></span> (회귀 예측값: <b>{st.session_state.avg_regression_pred:.2f}</b>)</p>", unsafe_allow_html=True)
            st.markdown("""
            <div style='font-size:25px;'>
            <b>당뇨망막병증 등급별 상세 설명:</b>
            <ul>
                <li><b>0등급(No DR):</b> 당뇨망막병증 없음</li>
                <li><b>1등급(Mild NPDR):</b> 경미한 비증식성 당뇨망막병증</li>
                <li><b>2등급(Moderate NPDR):</b> 중등도 비증식성 당뇨망막병증</li>
                <li><b>3등급(Severe NPDR):</b> 중증 비증식성 당뇨망막병증</li>
                <li><b>4등급(Proliferative DR)):</b> 증식성 당뇨망막병증</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("이미지를 업로드하고 '등급 분류 실행' 버튼을 클릭하여 예측을 시작하세요.")

# --- 탭 2: 판단 근거 시각화 및 소견 (기존 탭2 + 기존 탭3 통합) ---
with tab2:
    st.markdown("<h2 style='font-size: 3.5em; font-weight: bold;'>🔍 Sweet Vision의 판단 근거 시각화 및 소견</h2><br>", unsafe_allow_html=True)
    if st.session_state.raw_image_rgb_0_255 is None:
        st.warning("이미지를 업로드하면 Sweet Vision의 판단 근거를 시각화하고 소견을 확인할 수 있습니다.")
    elif st.session_state.pytorch_models is None:
        st.error("분류 모델이 로드되지 않아 시각화를 수행할 수 없습니다.")
    else:
        # 시각화 실행 전에 predicted_label_name이 있는지 확인
        if st.session_state.predicted_label_name is None:
            st.warning("시각화를 진행하려면 먼저 '📊 등급 분류' 탭에서 등급 분류를 실행해야 합니다.")
            st.info("등급 분류가 완료된 후 다시 이 탭으로 돌아와 '시각화 실행' 버튼을 눌러주세요.")
        else:
            # 시각화 실행 버튼
            if st.button("시각화 실행", key="visualize_button"):
                with st.spinner("AI가 분석 근거 이미지를 생성 중입니다..."):
                    image_for_model_pred = st.session_state.raw_image_rgb_0_255
                    image_for_model_pred_cropped = crop_image_to_circle(image_for_model_pred)
                    image_for_model_pred_preprocessed = ben_graham_preprocessing(image_for_model_pred_cropped)

                    transformed = pytorch_val_transforms(image=image_for_model_pred_preprocessed)
                    img_tensor_normalized = transformed['image'].unsqueeze(0).to(device)

                    # Calculate original_img_0_1_rgb
                    mean_rgb = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
                    std_rgb = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
                    st.session_state.original_img_0_1_rgb = img_tensor_normalized.squeeze(0).cpu().numpy().transpose((1, 2, 0))
                    st.session_state.original_img_0_1_rgb = st.session_state.original_img_0_1_rgb * std_rgb + mean_rgb
                    st.session_state.original_img_0_1_rgb = np.clip(st.session_state.original_img_0_1_rgb, 0, 1)

                    target_layer_name_for_gradcam = 'blocks_last_block'
                    
                    if st.session_state.pytorch_models is None or not st.session_state.pytorch_models:
                        st.error("AI 모델이 로드되지 않아 시각화를 생성할 수 없습니다. 관리자에게 문의하세요.")
                        st.session_state.cam_heatmap = None
                    else:
                        st.session_state.cam_heatmap = make_gradcam_heatmap_pytorch_ensemble(
                            img_tensor_normalized=img_tensor_normalized,
                            models=st.session_state.pytorch_models,
                            target_layer_name=target_layer_name_for_gradcam
                        )
                st.success("분석 근거 이미지 생성이 완료되었습니다.")

            # 시각화 결과 표시
            if (st.session_state.cam_heatmap is not None and
                np.max(st.session_state.cam_heatmap) > 0 and
                st.session_state.predicted_label_name is not None and
                st.session_state.original_img_0_1_rgb is not None):

                st.markdown("""
                <div style='font-size:25px;'>
                <b>시각화 설명</b>: Sweet Vision이 당뇨망막병증을 예측할 때 <b>주목한 영역을 색상으로 표시</b>합니다.
                <ul>
                    <li><b><span style='color: red;'>붉은 계열 (붉은색, 주황색, 노란색)</span></b>: AI가 병변으로 의심하고 높은 중요도를 부여한 영역입니다. 병변 존재 가능성이 높습니다.</li>
                    <li><b><span style='color: blue;'>푸른 계열 (푸른색, 녹색)</span></b>: AI가 병변과 관련성이 낮다고 판단한 영역입니다.</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                fig_xai = display_xai_results(
                    original_img_0_1_rgb=st.session_state.original_img_0_1_rgb,
                    gradcam_heatmap=st.session_state.cam_heatmap,
                    predicted_label_name=st.session_state.predicted_label_name
                )
                
                if fig_xai:
                    st.pyplot(fig_xai)
                else:
                    st.error("XAI 시각화 이미지를 생성할 수 없습니다. (matplotlib Figure 객체 문제)")
                
                # --- Sweet Vision의 이미지 소견 기능 통합 ---
                st.markdown("---") # 시각화와 소견 사이에 구분선 추가
                st.markdown("<h3 style='font-size:2.5em; font-weight: bold;'>📝 Sweet Vision의 이미지 분석 소견</h3>", unsafe_allow_html=True)
                
                if st.session_state.medgemma_pipe is None:
                    st.warning("MedGemma 파이프라인이 로드되지 않았습니다. Sweet Vision의 소견 기능을 사용할 수 없습니다.")
                else:
                    if st.button("Sweet Vision의 이미지 소견 생성", key="generate_llm_report_button_tab2"):
                        with st.spinner("Sweet Vision이 판독지를 생성 중입니다... (최대 30초 소요)"):
                            overlay_image_for_llm = create_gradcam_overlay_image_for_llm(
                                background_image_rgb=(st.session_state.original_img_0_1_rgb * 255).astype(np.uint8),
                                heatmap=st.session_state.cam_heatmap,
                                alpha=0.4
                            )

                            medgemma_report_text = generate_medgemma_report(
                                medgemma_pipe=st.session_state.medgemma_pipe,
                                predicted_label_name=st.session_state.predicted_label_name,
                                avg_regression_pred=st.session_state.avg_regression_pred,
                                overlay_image_for_llm=overlay_image_for_llm, # PIL Image 객체 전달
                                predicted_grade_num=predicted_grade_num, # 이 부분 추가
                                predicted_grade_english=predicted_grade_english # 이 부분 추가
                            )
                            st.session_state.medgemma_report_text = medgemma_report_text # 세션 상태에 저장

                        if st.session_state.medgemma_report_text:
                            st.markdown(f"<div style='font-size:25px;'>{st.session_state.medgemma_report_text}</div>", unsafe_allow_html=True)
                            st.success("✅ Sweet Vision 판독지 생성이 완료되었습니다.")
                        else:
                            st.error("❌ Sweet Vision 판독지 생성에 실패했습니다. MedGemma 모델 응답을 확인하세요.")
                    elif st.session_state.medgemma_report_text: # 버튼을 누르지 않았지만 이전에 생성된 소견이 있다면 표시
                        st.markdown(f"<div style='font-size:25px;'>{st.session_state.medgemma_report_text}</div>", unsafe_allow_html=True)
                        st.info("이전에 생성된 소견입니다. 다시 생성하려면 'Sweet Vision의 이미지 분석' 버튼을 클릭하세요.")
                    else: # 시각화는 됐지만 소견 버튼은 아직 안 누른 경우
                         st.info("상단의 'Sweet Vision의 이미지 분석' 버튼을 클릭하여 Sweet Vision이 분석한 전문 판독지를 받아보세요.")
            else:
                if st.session_state.raw_image_rgb_0_255 is not None and \
                   st.session_state.predicted_label_name is not None:
                    st.info("상단의 '시각화 실행' 버튼을 클릭하여 Sweet Vision의 판단 근거 이미지를 생성하고 소견을 받아보세요.")

# 불필요해진 탭3 제거 (주석 처리 또는 삭제)
# with tab3:
#     pass
