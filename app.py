import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document, StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

st.set_page_config(page_title="Chat with IIT Roorkee Noticeboard", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

openai.api_key = st.secrets.openai_key
st.title("Chat with IIT Roorkee Noticeboard 💬")

st.markdown("""This app is build on top of IIT Roorkee Noticeboard""")
    
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about IIT Roorkee Noticeboard!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data(use_cache=True):
    st.session_state.service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert in answering questions about IIT Roorkee Noticeboard. Be polite and helpful. Always use context to answer the question. If you don't know the answer to some question, just say so"))
    if use_cache:
        last_modified = "24-10-2023"
        with st.spinner(text=f"Using cached notices. Last updated on {last_modified}! This should take a few seconds."):
            storage_context = StorageContext.from_defaults(persist_dir="./storage")
            index = load_index_from_storage(storage_context)
            return index
    else:
        with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
            reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
            docs = reader.load_data()
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
            return index

index = load_data(use_cache=True)
# chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, system_prompt="You are an expert in answering questions about IIT Roorkee Noticeboard. Be polite and helpful. Always use context to answer the question. If you don't know the answer to some question, just say so")

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine =  index.as_chat_engine(chat_mode="condense_question", verbose=True, system_prompt="You are an expert in answering questions about IIT Roorkee Noticeboard. Be polite and helpful. Always use context to answer the question. If you don't know the answer to some question, just say so")

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history