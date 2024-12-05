# 공감형 챗봇 프로젝트

## 폴더소개
- EDA: 데이터 eda
  - 설명, 재밌는 인사이트 쓰기
  
- evalutation: 공감 대화 평가
- 
평가기준 1.
같은 input을 넣었을 때, 원 데이터셋의 output 의 category와 우리의 output이 받는 카테고리가 일치하는가?
평가기준 2.
기존의 output과 우리가 뱉은 output의 임베딩 거리는 얼마나 비슷한가?
평가기준 3. (human)
실제로 사람들의 경험이 긍정적인 쪽으로 변화하였는가?
사람들이 대답을 얼마나 사람답다고 느끼는가?

- model_training:
  모델 학습용 코드 
  model_training/models = config, weight, model structure
  model_training/results_ = 모델 훈련 도중 일정 에폭마다 저

- 나머지 나와있는 코드: 학습용 코드 (찐)
- inference_test: 하이퍼 파라미터를 일정 간격으로 변경 (그리드 서치)해서 모델과 텍스트를 넣어서 답변을 생성해서 일일이 확인 => 잘 나오는 답뱐ㅇ; 나오는 파라미터를 찾으려

- docker.yaml: docker setting 
