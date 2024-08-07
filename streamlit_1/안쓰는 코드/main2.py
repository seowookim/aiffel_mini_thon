# 영상을 참고해서 만들기 시작 https://www.youtube.com/watch?v=ZVmLe3odQvc

import streamlit as st
# from langchain_openai import OpenAI
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


st.set_page_config(page_title="공감챗", page_icon="💬",
                   layout="wide", initial_sidebar_state="expanded")
st.title("💬공감 챗봇이걸랑~")

# 세션 저장하는 리스트 선언
if "messages" not in st.session_state:
    st.session_state.messages = []

# 저장된 모델 및 토크나이저 로드
# KoGPT2 모델 로드
trained_model = GPT2LMHeadModel.from_pretrained('./kogpt2-chatbot')
trained_tokenizer = PreTrainedTokenizerFast.from_pretrained('./kogpt2-chatbot')
# trained_model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
# trained_tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2')

# 이전 대화기록을 출력해주는 코드
def print_message():
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        for chat_message in st.session_state.messages:
            st.chat_message(chat_message.role).write(chat_message.content)
print_message()


if user_input := st.chat_input("무엇이 궁금하신가요?"):
    # 사용자가 입력한 내용
    st.chat_message("user").write(f"{user_input}")
    st.session_state.messages.append(ChatMessage(role="user", content=user_input))
    
    # pre-trained model을 사용해서 답변 생성
    # 입력 문장 토큰화
    # user_input = "안녕"
    input_ids = trained_tokenizer.encode(trained_tokenizer.bos_token + user_input + trained_tokenizer.eos_token, return_tensors='pt')

    # 모델 추론
    outputs = trained_model.generate(input_ids, max_length=50, repetition_penalty=2.0, num_beams=5, early_stopping=True)
    msg = trained_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(msg)
    
    # AI의 답변
    with st.chat_message("assistant"):
        # msg = f"안녕하세요! {user_input}에 대해 알려드릴게요!"
        st.write(msg)
        st.session_state.messages.append(ChatMessage(role="assistant", content=msg))