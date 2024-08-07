# 영상을 참고해서 만들기 시작 https://www.youtube.com/watch?v=ZVmLe3odQvc

import streamlit as st
from transformers import PreTrainedTokenizerFast, AutoModelForSequenceClassification, GPT2LMHeadModel
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import torch
from peft import LoraConfig, TaskType, get_peft_model

# ---------------------------------------------------------------- #
#%% Streamlit 앱

st.set_page_config(page_title="공감챗", page_icon="💬")
st.title("💬공감 챗봇이걸랑~")

# 세션 저장하는 리스트 선언
if "messages" not in st.session_state:
    st.session_state.messages = []

# 채팅 기록을 저장하는 세션상태 변수
if "store" not in st.session_state:
    st.session_state.store = dict()
    
with st.sidebar:
    session_id = st.text_input("이름을 알려주세요", value="ex) 워라밸")
    
    clear_btn = st.button("대화 초기화")
    if clear_btn:
        st.session_state.messages = []
        st.session_state.store = dict()

# 이전 대화기록을 출력해주는 코드
def print_message():
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        for chat_message in st.session_state.messages:
            st.chat_message(chat_message.role).write(chat_message.content)
print_message()

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:  # 세션 ID가 store에 없는 경우
        st.session_state.store[session_id] = ChatMessageHistory() # 새로운 ChatMessageHistory 객체를 생성하고 store에 저장
    session_history = st.session_state.store[session_id]

    # 메시지 전체 텍스트 길이가 n자를 넘으면 가장 오래된 메시지를 제거
    total_length = sum(len(m.content) for m in session_history.messages)
    while total_length > 500 and session_history.messages:
        total_length -= len(session_history.messages[0].content)
        session_history.messages.pop(0)

    return session_history  # 세션 ID에 해당하는 ChatMessageHistory 객체 반환

# ---------------------------------------------------------------- #
#%% 추론 모델 준비

# 변수
# 그 외 추가 가능 인자 https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
# threshold=0.55 # float, 0~1
# max_length = 45 # int, default=20
# min_new_tokens = 3 # int
# use_cache = True # bool, default=True
# repetition_penalty = 4.0 # float, default=1.0
# do_sample = False # bool, default=False
# num_beams = 2 # int, default=1
# temperature = 2.0 # float, default=1.0
# top_k = 50 # int, default=50
# top_p = 0.9 # float, default=1.0

# 변수들을 sidebar에서 입력받기
with st.sidebar:
    threshold = st.slider("Threshold", 0.0, 1.0, 0.55)
    max_length = st.number_input("Max Length", value=45, step=1)
    min_new_tokens = st.number_input("Min New Tokens", value=3, step=1)
    use_cache = st.checkbox("Use Cache", value=True)
    repetition_penalty = st.number_input("Repetition Penalty", value=4.0, step=0.1)
    do_sample = st.checkbox("Do Sample", value=False)
    num_beams = st.number_input("Num Beams", value=2, step=1)
    temperature = st.number_input("Temperature", value=2.0, step=0.1)
    top_k = st.number_input("Top K", value=50, step=1)
    top_p = st.number_input("Top P", value=0.9, step=0.1)

cls_peft_config = LoraConfig(
    task_type="SEQ_CLS",
    inference_mode=True,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none")

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
SENT = '<unused1>'
PAD = '<pad>'

# 저장된 모델 및 토크나이저 로드
cls_path = './models/kogpt2-classification-lora'
cls_model = AutoModelForSequenceClassification.from_pretrained(
      cls_path,
      num_labels=5,
      problem_type="multi_label_classification"
)
trained_cls_model = get_peft_model(cls_model, cls_peft_config)
trained_cls_tokenizer = PreTrainedTokenizerFast.from_pretrained(cls_path,
                          bos_token=BOS, eos_token=EOS, unk_token=UNK,
                          pad_token=PAD, mask_token=MASK)

def predict_listener_empathy(input_text, model, tokenizer, threshold=threshold):
    # 모델을 평가 모드로 전환
    model.eval()

    # 입력 문장 토큰화
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True, max_length=128)

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

# 저장된 모델 및 토크나이저 로드
gen_path = './models/kogpt2-chatbot'
gen_model = GPT2LMHeadModel.from_pretrained(gen_path)

trained_gen_model = get_peft_model(gen_model, gen_peft_config)
trained_gen_tokenizer = PreTrainedTokenizerFast.from_pretrained(gen_path,
                          bos_token=BOS, eos_token=EOS, unk_token=UNK,
                          pad_token=PAD, mask_token=MASK)

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

    # 모델 추론
    outputs = model.generate(input_ids,
                             max_length=max_length, min_new_tokens=min_new_tokens, use_cache=use_cache,
                             repetition_penalty=repetition_penalty, do_sample=do_sample, num_beams=num_beams,
                             temperature=temperature, top_k=top_k, top_p=top_p, early_stopping=True)
    output_text = trained_gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text = output_text.split(A_TKN)[1]

    return output_text

# ---------------------------------------------------------------- #
#%% 챗봇 대화 로직

# if user_input := st.chat_input("무엇이 궁금하신가요?"):
#     # 사용자가 입력한 내용
#     st.chat_message("user").write(f"{user_input}")
#     st.session_state.messages.append(ChatMessage(role="user", content=user_input))
    
#     # 분류 결과 추론
#     # threshold 잘 설정해야
#     predicted_labels = predict_listener_empathy(user_input, trained_cls_model, trained_cls_tokenizer, threshold)

#     # 대화 기록을 포함한 전체 프롬프트 생성
#     # session_history = get_session_history(session_id)
#     # history_messages = [ChatMessage(role=m.role, content=m.content) for m in session_history.messages]
#     # history_messages.append(ChatMessage(role="user", content=user_input))
#     # print("history_messages", history_messages)

#     # 응답 생성
#     msg = predict_answer(predicted_labels, user_input, trained_gen_model, trained_gen_tokenizer,
#                    max_length, min_new_tokens, use_cache,
#                    repetition_penalty, do_sample, num_beams,
#                    temperature, top_k, top_p)
#     print("msg", msg)

#     # AI의 답변
#     with st.chat_message("assistant"):
#         st.write(msg)
#         st.session_state.messages.append(ChatMessage(role="assistant", content=msg))
#         # 세션 기록에 AI의 응답 추가
#         # session_history.add_message(ChatMessage(role="assistant", content=msg))
#         # print("history_messages", session_history)

# 챗봇 대화 로직에서 함수 호출 부분
if user_input := st.chat_input("무엇이 궁금하신가요?"):
    # 사용자가 입력한 내용
    st.chat_message("user").write(f"{user_input}")
    st.session_state.messages.append(ChatMessage(role="user", content=user_input))
    
    # 분류 결과 추론
    predicted_labels = predict_listener_empathy(user_input, trained_cls_model, trained_cls_tokenizer, threshold)

    # 응답 생성
    msg = predict_answer(predicted_labels, user_input, trained_gen_model, trained_gen_tokenizer,
                         max_length, min_new_tokens, use_cache,
                         repetition_penalty, do_sample, num_beams,
                         temperature, top_k, top_p)

    # AI의 답변
    with st.chat_message("assistant"):
        st.write(msg)
        st.session_state.messages.append(ChatMessage(role="assistant", content=msg))
    