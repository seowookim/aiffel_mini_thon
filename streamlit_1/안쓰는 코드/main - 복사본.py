# 영상을 참고해서 만들기 시작 https://www.youtube.com/watch?v=ZVmLe3odQvc

import streamlit as st
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks.base import BaseCallbackHandler


st.set_page_config(page_title="공감챗", page_icon="💬",
                   layout="wide", initial_sidebar_state="expanded")
st.title("💬공감 챗봇이걸랑~")

# 세션 저장하는 리스트 선언
if "messages" not in st.session_state:
    st.session_state.messages = []

# 채팅 기록을 저장하는 세션상태 변수
if "store" not in st.session_state:
    st.session_state.store = dict()

# 저장된 모델 및 토크나이저 로드
# KoGPT2 모델 로드
trained_model = GPT2LMHeadModel.from_pretrained('./kogpt2-chatbot')
trained_tokenizer = PreTrainedTokenizerFast.from_pretrained('./kogpt2-chatbot')

# 이전 대화기록을 출력해주는 코드
def print_message():
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        for chat_message in st.session_state.messages:
            st.chat_message(chat_message.role).write(chat_message.content)
print_message()

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store: # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하고 store에 저장
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id] # 세션 ID에 해당하는 ChatMessageHistory 객체 반환

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initail_text=""):
        self.container = container
        self.initail_text = initail_text
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def generate_response(model, tokenizer, prompt_text):
    # 입력 문장 토큰화
    input_ids = tokenizer.encode(tokenizer.bos_token + prompt_text + tokenizer.eos_token, return_tensors='pt')

    # 모델 추론
    outputs = model.generate(input_ids, max_length=50, repetition_penalty=2.0, num_beams=5, early_stopping=True)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text

if user_input := st.chat_input("무엇이 궁금하신가요?"):
    # 사용자가 입력한 내용
    st.chat_message("user").write(f"{user_input}")
    st.session_state.messages.append(ChatMessage(role="user", content=user_input))
    
    # pre-trained model을 사용해서 답변 생성
    #  프롶프트 생성
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "질문에 공감 하는 방식으로 응답해주세요" ,
            ) ,
            # 대화 기록을 함
            MessagesPlaceholder(variable_name="history"),
            ('human',"{question}"), # 사용자의 질문을 입력 
        ]
    )
    
    # 대화 기록을 포함한 전체 프롬프트 생성
    session_history = get_session_history("122")
    history_messages = [ChatMessage(role=m.role, content=m.content) for m in session_history.messages]
    full_prompt = prompt.format(history=history_messages, question=user_input)
    
    # 응답 생성
    msg = generate_response(trained_model, trained_tokenizer, full_prompt)

    # AI의 답변
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        st.write(msg)
        st.session_state.messages.append(ChatMessage(role="assistant", content=msg))
        # 세션 기록에 AI의 응답 추가
        session_history.add_message(ChatMessage(role="assistant", content=msg))
        
    