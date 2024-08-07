# 라이브러리 로드
import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import wandb
from peft import LoraConfig, TaskType, get_peft_model
import evaluate
import torch
import gc
import os

#%% 변수 설정
modelNm = 'kogpt2'
epoch = 10 # 학습 횟수
max_len = 128 # 문장 최대 길이

# sweep 이름
sweep_name = f'{modelNm}_epoch{epoch}_maxlen{max_len}'

#%% 데이터 로드
train_df_org = pd.read_csv('./data/train_data.csv')
valid_df_org = pd.read_csv('./data/validation_data.csv')

#%% 원본 데이터에서 필요한 데이터셋으로 변경
def extract_data(df):
  filtered_text = []
  filtered_empathy = []

  # 데이터프레임을 순회하며 조건에 맞는 데이터 추출
  for i in tqdm(range(len(df)-1)):
      if df.loc[i, 'speaker'] == 0 and df.loc[i + 1, 'speaker'] == 1 and df.loc[i + 1, 'empathy'] != 0:
          filtered_text.append(df.loc[i, 'text'])
          filtered_empathy.append(df.loc[i + 1, 'empathy'])

  # 결과를 데이터프레임으로 생성
  return pd.DataFrame({'text': filtered_text, 'empathy': filtered_empathy})

train_df = extract_data(train_df_org)
valid_df = extract_data(valid_df_org)

print(len(train_df), len(valid_df))

#%% 토큰화
Q_TKN = "<Q>"
A_TKN = "<A>"
BOS = '</s>'
EOS = '</s>'
UNK = '<unk>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

# KoGPT2 토크나이저 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token=UNK,
            pad_token=PAD, mask_token=MASK)

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

label_classes = ['1', '2', '3', '4', '5']
mlb = MultiLabelBinarizer(classes=label_classes)

#%% 데이터셋 전처리
def get_cls_input(examples):
    inputs = examples['text']
    labels = [label.split(',') for label in examples['empathy']]
    model_inputs = tokenizer(inputs, max_length=max_len, truncation=True, padding="max_length")

    labels_binary = mlb.fit_transform(labels)
    labels = labels_binary.astype(np.float32).tolist()

    model_inputs["labels"] = labels
    return model_inputs

train_inputs = train_dataset.map(get_cls_input, batched=True)
valid_inputs = valid_dataset.map(get_cls_input, batched=True)

print(train_inputs, valid_inputs)

#%% 모델 학습을 위한 하이퍼 파라미터 세팅
wandb.login()

# hyperparameters
parameters_dict = {
    'lr_scheduler_type': {
        'values': ['linear', 'cosine', 'polynomial']  # 학습률 스케줄러의 타입을 선택 가능한 값들로 설정
    },
    'learning_rate': {
        'distribution': 'log_uniform_values',  # 학습률을 로그 스케일로 균등하게 분포시켜 선택
        'min': 1e-5,  # 학습률의 최소값
        'max': 1e-3  # 학습률의 최대값
    },
    'weight_decay': {
        'values': [0.1, 0.3, 0.5]  # 가중치 감쇠 값을 선택 가능한 값들로 설정
    },
    'train_batch_size': {
        'values': [8, 16, 32]  # 학습 시 배치 크기를 선택 가능한 값들로 설정
    },
    'eval_batch_size': {
        'values': [8, 16, 32]  # 평가 시 배치 크기를 선택 가능한 값들로 설정
    }
}

# method
sweep_config = {
    'name': sweep_name,
    'method': 'bayes',  # 베이지안 최적화 방법
    'metric': {
        'name': 'eval_loss',
        'goal': 'minimize'
    },
    'parameters': parameters_dict  # 기존의 parameters_dict 사용
}

# 로라 적용
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none")

# metric 설정
def compute_metrics(pred):
    logits, labels = pred
    predictions = (logits > 0).astype(float)  # 시그모이드 활성화 함수 적용 후 임계값 0.5 기준으로 이진
    labels = labels.flatten()
    predictions = predictions.flatten()

    # 정확도, F1 score 계산
    acc_metrics = evaluate.load("accuracy")
    accuracy = acc_metrics.compute(predictions=predictions, references=labels)

    f1_metrics = evaluate.load("f1")
    f1 = f1_metrics.compute(predictions=predictions, references=labels)

    return {"accuracy": accuracy, "f1_metrics": f1}

def train():
  wandb.init()
  config = wandb.config

  # KoGPT2 모델 로드
  model = AutoModelForSequenceClassification.from_pretrained(
      "skt/kogpt2-base-v2",
      num_labels=5,
      problem_type="multi_label_classification"
  )
  # lora 적용
  model = get_peft_model(model, peft_config)

  # 학습 설정
  total_train_steps = (len(train_dataset) // config.train_batch_size) * epoch
  save_turn = total_train_steps // (epoch / 0.25)  # 0.25 에포크마다 저장
  log_turn = save_turn // 1  # 특정 에포크마다 평가하여 wanDB에 기록, 평가와 저장은 배수여야 함

  training_args = TrainingArguments(
      fp16=True,
      output_dir=f'./results_{modelNm}_cls',
      num_train_epochs=epoch,
      lr_scheduler_type=config.lr_scheduler_type,
      learning_rate=config.learning_rate,
      per_device_train_batch_size=config.train_batch_size,
      per_device_eval_batch_size=config.eval_batch_size,
      warmup_steps=500,
      weight_decay=config.weight_decay,
      logging_dir='./logs',
      logging_steps=log_turn,  # 로그를 기록할 스텝 간격
      do_eval=True,
      evaluation_strategy="steps",
      eval_steps=log_turn,  # 평가를 할 스텝 간격
      remove_unused_columns=True,
      save_steps=save_turn,  # 모델을 저장할 스텝 간격
    #   save_total_limit=3,  # 저장할 체크포인트의 최대 개수
      load_best_model_at_end=True,  # 학습 종료 시 최고의 모델을 로드
      metric_for_best_model="eval_loss"  # 최적 모델을 선택할 기준 메트릭
  )

  # 체크포인트 디렉토리 설정
  last_checkpoint = None
  if os.path.isdir(training_args.output_dir) and os.listdir(training_args.output_dir):
      checkpoint_files = [f for f in os.listdir(training_args.output_dir) if f.startswith('checkpoint')]
      if checkpoint_files:
          last_checkpoint = os.path.join(training_args.output_dir, checkpoint_files[-1])

  # Trainer 설정
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_inputs,
      eval_dataset=valid_inputs,
      compute_metrics=compute_metrics,
  )

  # 학습
  trainer.train(resume_from_checkpoint=last_checkpoint) # 중간에 멈춘 학습을 이어서 재개할 수 있도록 설정

  # 최적의 모델 저장
  best_model = trainer.model
  best_model.save_pretrained('./models/kogpt2-classification')
  tokenizer.save_pretrained('./models/kogpt2-classification')

# 캐쉬 삭제
gc.collect()
torch.cuda.empty_cache()

# 학습 실행
sweep_id = wandb.sweep(sweep_config, entity='nkim123', project='minidlthon_kogpt2_classification')
wandb.agent(sweep_id, train, count=1)
wandb.finish()

