# 🩺 SweetVision: AI 기반 안저 영상 당뇨망막병증( DR ) 등급 분류 솔루션

<p align="center">
  <img src="images/logo.png" alt="SweetVision Banner" width="200"/>
</p>

**SweetVision** 은 당뇨망막병증(Diabetic Retinopathy, DR) 조기 발견 및 진단 보조를 목표로 한  
**AI 기반 안저 영상 자동 판독 및 등급 분류 솔루션**입니다.  

- **문제 인식**:  
  - 국내 당뇨 환자 급증 (2010년 312만명 → 2020년 600만명)  
  - DR 유병률 최대 80%, 반복적·시간 소모적 수작업 판독 부담  
- **솔루션 목표**:  
  - CNN 기반 DR 자동 등급 분류  
  - Grad-CAM 기반 설명 가능한 AI (XAI) 시각화 제공  
  - MedGemma 멀티모달 LLM을 통한 임상적 수준 판독지 자동 생성  

---

## ⚙️ 기술 스택

| 구분       | 사용 기술 |
|-----------|-----------|
| 언어       | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) |
| 딥러닝 모델 | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![EfficientNet](https://img.shields.io/badge/EfficientNet--B5-2E8B57?style=flat) |
| XAI        | ![Grad-CAM](https://img.shields.io/badge/Grad--CAM-4682B4?style=flat) |
| LLM        | ![MedGemma](https://img.shields.io/badge/MedGemma-8A2BE2?style=flat) |
| 프레임워크/플랫폼 | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) |
| 협업 도구   | ![Slack](https://img.shields.io/badge/Slack-4A154B?style=flat&logo=slack&logoColor=white) ![GoogleDrive](https://img.shields.io/badge/Google_Drive-4285F4?style=flat&logo=googledrive&logoColor=white) |

---

## 🏗 시스템 아키텍처

![아키텍처](images/sweetvision_architecture.png)

---

## 🔑 주요 기능

### 1. 안저 이미지 입력 & 전처리
> 안구 ROI 추출, Ben Graham 기법으로 병변 특징 강조 → 데이터 증강·정규화 후 모델 입력

---

### 2. DR 등급 분류 (0~4단계)
> **EfficientNet-B5 기반 CNN 모델**로 당뇨망막병증 등급 자동 분류 (Kappa score 0.92 달성)

---

### 3. 설명 가능한 AI (XAI) 시각화
> **Grad-CAM**으로 모델이 주목한 병변 위치를 히트맵으로 시각화, 판독 신뢰도 확보

---

### 4. 임상 판독지 자동 생성
> **MedGemma 멀티모달 LLM**으로 시각화 이미지 + 예측 결과 기반 임상 수준 판독지 자동 생성  

---

## 🌟 기대효과
- 전문의 판독 보조 → 반복적 판독 업무 경감  
- 조기 진단 통한 실명 예방 및 장기적 의료비 절감  
- EMR·PACS 연동으로 병원 진료 효율 향상  
- 원격의료·공공의료 환경에서도 활용 가능  

---

## 👥 핵심 기여
- **데이터 전처리**: ROI 추출, Ben Graham 대비 강화, 증강 파이프라인 설계  
- **모델 개발**: EfficientNet-B5 기반 CNN 모델 학습 및 3-Fold 교차검증 적용  
- **성능 개선**: 클래스 불균형 가중치, Test-Time Augmentation, Optimized Rounder 적용 → Kappa 0.92 달성  
- **XAI + LLM 통합**: Grad-CAM 시각화와 MedGemma 기반 프롬프트 설계로 임상적 판독 리포트 자동 생성  

---

## 📂 폴더 구조 (예시)

<pre>
📂 SweetVision/
 ┣ 📜 app.py                  # Streamlit 메인 실행 파일
 ┣ 📜 model.py                # EfficientNet-B5 모델 정의 및 학습 코드
 ┣ 📜 preprocess.py           # 이미지 전처리 (ROI 추출, Ben Graham, 증강)
 ┣ 📜 xai.py                  # Grad-CAM 기반 시각화 모듈
 ┣ 📜 llm_prompt.py           # MedGemma 프롬프트 엔지니어링
 ┣ 📂 data/                   # 학습/테스트용 이미지 데이터
 ┣ 📂 outputs/                # 결과 저장 (히트맵, 판독 리포트)
 ┗ 📂 images/                 # README 및 UI 스크린샷
</pre>
