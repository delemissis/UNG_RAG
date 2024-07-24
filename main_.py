import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import streamlit as st
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import sqlite3

os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]

st.title("Údarás na Gaeltachta")
# Default language is Irish
language = "Irish"

# Translation of static text based on language selection
language_text = "Select language" if language == "English" else "Roghnaigh teanga"

# Language selection toggle
language = st.sidebar.radio(language_text, ("Irish", "English"))

# Define the translation functions
def irish_to_english(user_input: str) -> str:
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Translate the following text from Irish to English: {text}"
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    response = llm_chain.run(user_input)
    return response

def english_to_irish(model_response: str) -> str:
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Translate the following text from English to Irish: {text}"
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    response = llm_chain.run(model_response)
    return response

# Translation of static text based on language selection
ask_question_text = "Ask me a question" if language == "English" else "Cuir ceist orm"
curious_about_text = "What do you want to ask?" if language == "English" else "Cad ba mhaith leat a iarraidh?"

with st.chat_message("assistant"):
    st.text(ask_question_text)

# Initializing message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Displaying the message history on re-run
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if 'question' in message:
            st.markdown(message["question"])
        elif 'response' in message:
            st.write(message['response'])
        elif 'error' in message:
            st.text(message['error'])

# Getting the questions from the users
user_question = st.chat_input(curious_about_text)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the NEH docs – hang tight! This should take 1-2 minutes."):
        loader = RecursiveUrlLoader("https://www.neh.gov.ie/")
        docs = loader.load()
        # print(docs[0].metadata)

        # print(docs[1].metadata)

        # print(docs[5].metadata)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()

        return retriever

retriever = load_data()

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}

# @st.cache_data
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# @st.cache_data
def handle_query(user_question, language):

    if language == "Irish":
        user_question = irish_to_english(user_question)
        print(user_question)
        
    response = conversational_rag_chain.invoke(
        {"input": user_question},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in store.
    )["answer"]

    if language == "Irish":
        response = english_to_irish(response)
        print(response)

    return response

if user_question:
    # Displaying the user question in the chat message
    with st.chat_message("user"):
        st.markdown(user_question)
    # Adding user question to chat history
    st.session_state.messages.append({"role": "user", "question": user_question})

    try:
        with st.spinner("Analysing..."):
            response = handle_query(user_question, language)
            st.session_state.messages.append({"role": "assistant", "response": response})
            st.write(response)
    except Exception as e:
        st.write(e)
        error_message = "⚠️Sorry, Couldn't generate the answer! Please try rephrasing your question!"
        st.session_state.messages.append({"role": "assistant", "error": error_message})

# # Function to clear history
# def clear_chat_history():
#     st.session_state.messages = []

# # Button for clearing history
# st.sidebar.text("Click to Clear Chat history")
# st.sidebar.button("CLEAR 🗑️", on_click=clear_chat_history)
