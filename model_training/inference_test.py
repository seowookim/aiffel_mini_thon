import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import PreTrainedTokenizerFast, AutoModelForSequenceClassification, GPT2LMHeadModel
from accelerate import Accelerator
import itertools
import pandas as pd

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 변수
# 그 외 추가 가능 인자 https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
cls_path = './models/kogpt2-classification_lora_best' # 분류 모델 경로
gen_paths = ['./models/kogpt2_chatbot_no_lora_0.1', 
             './models/kogpt2_chatbot_no_lora_0.2',
             './models/kogpt2_chatbot_no_lora_0.3',
             './models/kogpt2_chatbot_no_lora_0.5',
             './models/kogpt2_chatbot_no_lora_0.7',
             './models/kogpt2_chatbot_no_lora_1',
             './models/kogpt2_chatbot_lora_best'
             ]  # 여러 생성 모델 경로

# 하이퍼파라미터 그리드
thresholds = [0.5, 0.7, 0.9]
max_length = 128
min_new_tokens_list = [2, 6]
repetition_penalties = [1.0, 2.0, 5.0]
do_samples = [True, False]
num_beams_list = [1, 5]
temperatures = [0.7, 1.0, 2.0]
top_ks = [50, 100]
top_ps = [0.9, 1.0]
use_cache = False

USE_LORA_CLS_MODEL = True
USE_LORA_GEN_MODEL = False

cls_peft_config = LoraConfig(
    task_type="SEQ_CLS",
    inference_mode=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none")
distil_cls_peft_config = LoraConfig(
    task_type = "SEQ_CLS", #시퀀스 분류 작업을 위한 설정
    inference_mode = True, #모델이 추론모드인지 여부 설정
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['attention.q_lin', 'attention.k_lin', 'attention.v_lin', 'attention.out_lin', 'ffn.lin1', 'ffn.lin2']
)

gen_peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none")

Q_TKN = "<Q>"
A_TKN = "<A>"
BOS = '</s>'
EOS = '</s>'
UNK = '<unk>'
MASK = '<unused0>'
SENT = '<sent>'
PAD = '<pad>'

# 저장된 모델 및 토크나이저 로드
cls_model = AutoModelForSequenceClassification.from_pretrained(
      cls_path,
      num_labels=5,
      problem_type="multi_label_classification"
)
if USE_LORA_CLS_MODEL:
  accelerator = Accelerator() #데이터 병렬 처리 원활
  cls_model = accelerator.prepare(cls_model)
  cls_model = get_peft_model(cls_model, cls_peft_config)
cls_tokenizer = PreTrainedTokenizerFast.from_pretrained(cls_path,
                        bos_token=BOS, eos_token=EOS, unk_token=UNK,
                        pad_token=PAD, mask_token=MASK)


def predict_listener_empathy(input_text, model, tokenizer, threshold, max_length=max_length):
    # 모델을 평가 모드로 전환
    model.eval()

    # 입력 문장 토큰화
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 입력 데이터를 장치로 이동
    # 모델에 입력을 전달하여 로짓(logits)을 얻음
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 로짓에 시그모이드 적용하여 확률로 변환
    probabilities = torch.sigmoid(logits)
    # 임계값을 기준으로 이진화
    predictions = (probabilities > threshold).int()

    # 레이블 디코딩
    label_classes = ['조언', '격려', '위로', '동조', '']
    num_classes = 5
    predicted_labels = [label_classes[i] for i in range(num_classes) if predictions[0][i] == 1]

    return predicted_labels

def predict_answer(predicted_labels, input_text, model, tokenizer,
                max_length, min_new_tokens, use_cache,
                repetition_penalty, do_sample, num_beams,
                temperature, top_k, top_p):
    # 모델을 평가 모드로 전환
    model.eval()
    # 입력 문장 토큰화
    empathy = ' ,'.join(map(str, predicted_labels))
    inputs = Q_TKN + input_text + SENT + empathy + A_TKN
    input_ids = tokenizer.encode(tokenizer.bos_token + inputs + tokenizer.eos_token, return_tensors='pt')
    input_ids = input_ids.to(device)  # 입력 데이터를 장치로 이동

    # 모델 추론
    outputs = model.generate(input_ids,
                            max_length=max_length, min_new_tokens=min_new_tokens, use_cache=use_cache,
                            repetition_penalty=repetition_penalty, do_sample=do_sample, num_beams=num_beams,
                            temperature=temperature, top_k=top_k, top_p=top_p, early_stopping=True)
    output_text = trained_gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text = output_text.split(A_TKN)[1]

    return output_text


# 결과를 저장할 리스트
results = []
batch_size = 10000
batch_number = 0
# 입력 문장
input_text = '너무 힘들어'

# 모든 하이퍼파라미터 조합을 반복
for gen_path in gen_paths:
    # 저장된 모델 및 토크나이저 로드
    gen_model = GPT2LMHeadModel.from_pretrained(gen_path)
    gen_model.to(device)  # 모델을 장치로 이동
    if USE_LORA_GEN_MODEL:
        accelerator = Accelerator() #데이터 병렬 처리 원활
        gen_model = accelerator.prepare(gen_model)
        gen_model = get_peft_model(gen_model, gen_peft_config)

    trained_gen_tokenizer = PreTrainedTokenizerFast.from_pretrained(gen_path,
                            bos_token=BOS, eos_token=EOS, unk_token=UNK,
                            pad_token=PAD, mask_token=MASK)

    for threshold in thresholds:
        predicted_labels = predict_listener_empathy(input_text, cls_model, cls_tokenizer, threshold)

        for i, (min_new_tokens, repetition_penalty, do_sample, num_beams, temperature, top_k, top_p) in enumerate(itertools.product(
                min_new_tokens_list, repetition_penalties, do_samples, num_beams_list, temperatures, top_ks, top_ps)):
            
            print(f'gen_path: {gen_path}, threshold: {threshold}, max_length: {max_length}, min_new_tokens: {min_new_tokens}, repetition_penalty: {repetition_penalty}, do_sample: {do_sample}, num_beams: {num_beams},temperature: {temperature}, top_k: {top_k}, top_p: {top_p}')
            
            
            response = predict_answer(predicted_labels, input_text, gen_model, trained_gen_tokenizer,
                            max_length, min_new_tokens, use_cache,
                            repetition_penalty, do_sample, num_beams,
                            temperature, top_k, top_p)

            # 결과 저장
            results.append({
                'input_text': input_text,
                'gen_path': gen_path.split('_', 2)[-1],
                'threshold': threshold,
                'max_length': max_length,
                'min_new_tokens': min_new_tokens,
                'repetition_penalty': repetition_penalty,
                'do_sample': do_sample,
                'num_beams': num_beams,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'predicted_labels': predicted_labels,
                'response': response
            })

            # 30번마다 저장
            if (i + 1) % batch_size == 0:
                batch_number += 1
                df = pd.DataFrame(results)
                file_name = f'./data/grid_search_results_batch_{batch_number}.csv'
                df.to_csv(file_name, index=False, encoding='utf-8-sig')
                results = []  # 저장 후 리스트 초기화

# 남은 결과 저장
if results:
    batch_number += 1
    df = pd.DataFrame(results)
    file_name = f'./data/grid_search_results_batch_{batch_number}.csv'
    df.to_csv(file_name, index=False, encoding='utf-8-sig')