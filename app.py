from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
# from custom import CustomLLM

# Create an instance of the CustomLLM class
# custom_llm = CustomLLM()
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import time

st.set_page_config(page_title="Probattle-05")
col1, col2, col3 = st.columns([1,4,1])
with col2:
    st.image("./icon.png",  width=50)

st.markdown(
    """
    <style>
div.stButton > button:first-child {
    background-color: #ffd0d0;
}
div.stButton > button:active {
    background-color: #ff6262;
}
   div[data-testid="stStatusWidget"] div button {
        display: none;
        }
    
    .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    button[title="View fullscreen"]{
    visibility: hidden;}
        </style>
""",
    unsafe_allow_html=True,
)

def reset_conversation():
  st.session_state.messages = []
  st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history",return_messages=True) 

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("ppc_vector_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity",search_kwargs={"k": 4})

prompt_template = """<s>[INST]This is a chat template and You are an IBA bot. Your task is to answer questions strictly based on the provided data below. If the answer is not available within this data, state that the information is not present in the document.
Do not mention that you are an AI Assistant, and do not say thank you. be concise as possible
 Use the following pieces of context to answer the question at the end.
 - If you think you need more information to find proper answer, ask user to clarify.
 - If you cannot find a proper answer, just say we don't have information about it.
 - Answer responsively and clearly, avoiding unnecessary punctuation. Use bullet points or numbered lists where appropriate for steps or structured information.- dont answer other then the provided context even dont say about other locations too always says according to ICAP or say According to ICAP.
 - Most important dont provide user any sources name and any sensitive information just say source lies on IBA website.
 - dont answer other then the information provided.
 Use Four five sentences maximum and keep the answer as concise as possible where it needed to be.
IMPORTANT: Always answer in the language English.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question', 'chat_history'])

# Create an instance of the CustomLLM class
# custom_llm = CustomLLM()
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="",  # if you prefer to pass api key in directly instaed of using env vars
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

input_prompt = st.chat_input("Say something")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role":"user","content":input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...",expanded=True):
            result = qa.invoke(input=input_prompt)

            message_placeholder = st.empty()

            full_response = "⚠️ **_Note: Information provided from the IBA._** \n\n\n"
        for chunk in result["answer"]:
            full_response+=chunk
            time.sleep(0.02)
            
            message_placeholder.markdown(full_response+" ▌")
        st.button('Reset All Chat 🗑️', on_click=reset_conversation)

    st.session_state.messages.append({"role":"assistant","content":result["answer"]})


