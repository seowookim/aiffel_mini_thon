import streamlit as st
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
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
    if session_id not in st.session_state.store:  # 세션 ID가 store에 없는 경우
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]  # 세션 ID에 해당하는 ChatMessageHistory 객체 반환

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.container.markdown(self.text)
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def stream_generate_response(model, tokenizer, prompt_text, stream_handler=None):
    input_ids = tokenizer.encode(tokenizer.bos_token + prompt_text + tokenizer.eos_token, return_tensors='pt')
    output_ids = input_ids.clone()

    for _ in range(50):  # 최대 토큰 수 설정
        outputs = model(input_ids=output_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        output_ids = torch.cat([output_ids, next_token_id], dim=-1)
        
        next_token = tokenizer.decode(next_token_id[0])
        
        # 스트리밍 핸들러가 있는 경우, 토큰을 하나씩 업데이트
        if stream_handler:
            stream_handler.on_llm_new_token(next_token)
        
        # 끝나는 토큰을 만났을 때 중단
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response_text

if user_input := st.chat_input("무엇이 궁금하신가요?"):
    st.chat_message("user").write(f"{user_input}")
    st.session_state.messages.append(ChatMessage(role="user", content=user_input))

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "질문에 공감하는 방식으로 응답해주세요"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    session_history = get_session_history("122")
    history_messages = [ChatMessage(role=m.role, content=m.content) for m in session_history.messages]
    full_prompt = prompt.format(history=history_messages, question=user_input)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        msg = stream_generate_response(trained_model, trained_tokenizer, full_prompt, stream_handler)
        
        st.session_state.messages.append(ChatMessage(role="assistant", content=msg))
        session_history.add_message(ChatMessage(role="assistant", content=msg))
