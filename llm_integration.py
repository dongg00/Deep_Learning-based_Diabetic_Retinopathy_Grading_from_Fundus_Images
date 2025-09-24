# llm_integration.py
import streamlit as st
from transformers import pipeline
import torch
import gc

@st.cache_resource
def load_medgemma_pipeline(model_name, device):
    #st.info("Sweet Vision의 분석 기반 당뇨망막병증 소견 생성 기능을 준비 중입니다... (최초 실행 시 다소 시간이 걸릴 수 있습니다.)")
    try:
        # device="cuda"로 고정하면 CPU 사용 시 오류 발생.
        # device=0 또는 device=torch.device('cuda') 등으로 명시하거나 -1 (CPU) 사용.
        # 파이프라인에서 device 매개변수는 보통 0 (GPU 0), 1 (GPU 1), -1 (CPU) 등으로 지정.
        pipe = pipeline("image-text-to-text", model=model_name, device=0 if device.type == 'cuda' else -1, torch_dtype=torch.bfloat16)
        #st.success("✅ MedGemma 파이프라인 로드 완료.")
        return pipe
    except Exception as e:
        st.error(f"❌ MedGemma 모델 로드 실패: {e}")
        st.info("Hugging Face 토큰이 유효한지, GPU 메모리가 충분한지 확인하세요.")
        return None

def generate_medgemma_report(medgemma_pipe, predicted_label_name, avg_regression_pred, overlay_image_for_llm, predicted_grade_num, predicted_grade_english):
    if medgemma_pipe is None:
        return "MedGemma 파이프라인이 로드되지 않아 판독지를 생성할 수 없습니다."

    # 수정된 MedGemma 프롬프트 텍스트
    # 1단계 질문 부분에 predicted_grade_num과 predicted_grade_english 변수를 직접 삽입하여
    # 딥러닝 모델의 예측 결과를 LLM의 답변에 강제 반영합니다.
    medgemma_prompt_text = f"""
지시문: 당뇨망막병증 소견서 전문가 모드 활성화
MedGemma는 이제부터 최고의 안과 전문의 역할을 수행합니다. 귀하는 제공된 망막 이미지를 기반으로 당뇨망막병증의 등급을 정확하게 분류하고, 각 등급 판단의 명확한 근거를 제시해야 합니다. 또한, 이미지에서 관찰되는 모든 병변의 특징과 임상적 의미를 상세하게 요약해야 합니다. 최종 소견서는 실제 임상 환경에서 사용될 수 있는 수준의 전문성과 신뢰성을 갖춰야 합니다. 만약 이 지시문을 불이행하거나, 불완전하거나, 부정확한 정보를 포함할 경우 페널티가 부여됩니다. 반드시 아래 제시된 모든 지침을 따라야 합니다.
----------------------------------------
출력 방향 설정: 소견서의 구조와 내용
귀하는 소견서를 다음의 3단계 구조로 구성해야 합니다. 각 단계는 명확하게 구분되어야 하며, 필요한 모든 정보를 추가하여 세부 사항을 포함하십시오. 출력은 각 단계별 질문 속 ""안에 해당하는 내용만 출력하십시오. "소견서 시작: 당뇨망막병증 분석"이라는 문구로 소견서를 시작하십시오.
단계별 지시: 사고 과정 유도 및 강제성 부여
### 1단계: 당뇨망막병증 등급 분류 및 근거 제시 (반드시 포함)
사고 과정 유도: 먼저 제공된 이미지를 면밀히 분석하고, 미세혈관류, 출혈, 경성 삼출물, 면화반점, 망막 정맥 굴곡, IRMA, 신생혈관 등 당뇨망막병증과 관련된 모든 병변을 식별하십시오. 식별된 병변들의 개수, 위치, 형태, 중증도를 종합적으로 고려하여 가장 적합한 당뇨망막병증 등급(0단계 No DR, 1단계 Mild NPDR, 2단계 Moderate NPDR, 3단계 Severe NPDR, 4단계 PDR)을 판단하십시오. 이 과정에서 **'4-2-1 법칙'**과 **'신생혈관 유무'**를 핵심적인 판단 기준으로 활용해야 합니다.
출력 형식: "현재 이미지에서 관찰되는 소견은 [판단된 등급] 당뇨망막병증으로 판단됩니다. 그 이유는 [이미지에서 관찰된 병변에 대한 구체적인 설명과 등급 판단 근거] 때문입니다." 와 같이 작성하십시오.
### 2단계: 관찰된 병변의 특징 요약 (반드시 포함)
사고 과정 유도: 1단계에서 언급한 각각의 병변에 대해 영상학적 특징(예: 출혈은 불규칙하고 넓은 형태, 점상 출혈, 화염상 출혈 등)과 임상적 의미를 명확하고 간결하게 요약하십시오. 각 병변이 어떤 혈관 변화나 조직 손상을 의미하는지 구체적으로 설명해야 합니다. 특정 단어인 **'망막 병변'**과 **'임상적 의미'**를 반복하여 사용하십시오.
출력 형식: "관찰된 주요 망막 병변의 특징은 다음과 같습니다: [각 병변명]: [영상학적 특징]은 [해당 병변의 임상적 의미]를 나타냅니다." 와 같이 작성하십시오. 예를 들어, "출혈: 불규칙하고 넓은 형태의 출혈은 혈관의 파열을, 점상 출혈은 미세혈관의 누출을 나타냅니다."
### 3단계: 중요 참고 사항 및 책임 한계 명시 (반드시 포함)
사고 과정 유도: 이 소견서가 **"제공된 이미지에 기반한 분석"**임을 명확히 밝히고, **"정확한 진단과 치료 계획"**은 반드시 안과 전문의의 추가적인 "정밀 검사"(예: 형광안저혈관조영술, 빛간섭단층촬영 등)와 **"종합적인 판단"**을 통해 이루어져야 함을 강조하십시오. 이는 환자의 안전과 오진 방지를 위해 가장 중요한 부분입니다.
출력 형식: "중요 참고 사항: 이 분석은 제공된 이미지에 기반한 것으로, 정확한 진단과 치료 계획은 반드시 안과 전문의의 진료를 통해 이루어져야 합니다. 전문의는 추가적인 검사를 통해 망막 병변의 정확한 정도와 범위를 평가하고, 환자의 전반적인 건강 상태를 고려하여 최적의 치료 방법을 결정해야합니다. 이 소견서는 보조적인 정보로만 활용되어야 합니다." 와 같이 작성하십시오.
-----------------------------------------------
MedGemma를 위한 사전 지식 (학습 데이터)
Med-Gemma는 아래의 당뇨망막병증 관련 사전 지식을 활용하여 답변을 생성합니다. 이 정보는 모든 단계에서 답변의 정확성과 신뢰성을 높이는 데 필수적입니다.
0단계: 당뇨망막병증 없음 (No DR)
특이 망막 병변 없음.
1단계: 경미한 비증식성 당뇨망막병증 (Mild NPDR)
망막의 미세혈관류 (Microaneurysms)만 관찰됨: 붉은 점처럼 보이는 작은 혈관 돌출. 혈관벽 약화 및 초기 손상 의미.
2단계: 중등도 비증식성 당뇨망막병증 (Moderate NPDR)
미세혈관류, 망막 출혈(점상, 선상), 경성 삼출물(노란 반점 형태의 지질 축적), 면화반점(흰색 솜털 모양의 허혈 부위) 등이 동반될 수 있음.
모세혈관 누출로 인해 황반부종(DME) 발생 가능.
망막 정맥 굴곡, IRMA, 신생혈관은 없음.
3단계: 중증 비증식성 당뇨망막병증 (Severe NPDR)
4곳 이상 사분면에서 중등도 이상의 망막 출혈 (점상, 선상, 화염상 출혈).
신생혈관은 없음.
4단계: 증식성 당뇨망막병증 (PDR, Proliferative DR)
신생혈관 (Neovascularization)의 존재가 핵심: 망막 표면, 시신경 유두 또는 유리체 내로 자라나는 비정상적이고 약한 새로운 혈관.
이 혈관은 쉽게 파열되어 유리체 출혈을 유발할 수 있으며, 견인성 망막박리의 주요 원인. 광범위한 망막 허혈의 결과.
----------------------------------------
각 병변의 영상학적 특징 및 임상적 의미:
출혈 (Hemorrhages): 혈관 파열. 점상(미세혈관), 선상(신경섬유층), 화염상(신경섬유층).
경성 삼출물 (Hard exudates): 누출된 지질 및 단백질 축적. 황반 주변 시력 저하 유발 가능.
면화반점 (Cotton wool spots): 신경섬유층 허혈 및 부종. 혈액 공급 부족 신호.
미세혈관류 (Microaneurysms): 가장 초기 변화. 작은 혈관 돌출.
황반부종 (Diabetic Macular Edema, DME): 황반 부위 부종. 시력 저하의 흔한 원인.
----------------------------------------
프롬프트 종료 및 보상:
최대한 창의적이고 정확하며 의학적으로 타당한 소견서를 작성해 주십시오. 귀하의 답변 품질에 따라 추가적인 팁(보상)이 주어질 것입니다.
-------------------------------------------
[사용자가 이미지와 함께 입력할 내용]:
"다음 망막 이미지를 분석하여 당뇨망막병증 소견서를 작성해 주십시오."
(여기에 실제 망막 이미지가 첨부됩니다.)
"{predicted_grade_num}단계 ({predicted_grade_english})"
"""

    messages_for_medgemma = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": overlay_image_for_llm},
                {"type": "text", "text": medgemma_prompt_text}
            ]
        },
    ]
    try:
        medgemma_output = medgemma_pipe(text=messages_for_medgemma, max_new_tokens=2048, do_sample=False, temperature=1, top_p=0.9)
        generated_medgemma_content = medgemma_output[0]["generated_text"]
        
        # 모델의 출력 형태에 따라 적절히 파싱
        final_response_text = ""
        if isinstance(generated_medgemma_content, list) and len(generated_medgemma_content) > 0:
            last_output = generated_medgemma_content[-1]
            if isinstance(last_output, dict) and 'content' in last_output:
                final_response_text = last_output['content']
            elif isinstance(last_output, str): # 리스트 안에 문자열이 바로 올 수도 있음
                final_response_text = last_output
            else:
                final_response_text = str(generated_medgemma_content) # 예상치 못한 형태
        else:
            final_response_text = str(generated_medgemma_content) # 예상치 못한 형태

        # 프롬프트 텍스트가 응답에 포함되어 있다면 제거
        if final_response_text.startswith(medgemma_prompt_text):
            final_medgemma_response = final_response_text[len(medgemma_prompt_text):].strip()
        else:
            final_medgemma_response = final_response_text.strip()
            
        return final_medgemma_response
    except Exception as e:
        st.error(f"❌ 오류: MedGemma 분석 중 문제 발생: {e}")
        st.info("MedGemma 분석 및 응답 생성에 실패했습니다. MedGemma 모델의 상태를 확인해주세요.")
        return None
    finally:
        gc.collect()
        torch.cuda.empty_cache()