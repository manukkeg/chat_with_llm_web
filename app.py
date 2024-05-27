import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import PyPDFLoader
import os

os.environ["OPENAI_API_KEY"] = 

st.set_page_config(page_title = "LLM for chat with websites", page_icon="🌐")
st.title("Chat with websites")

documents_chunks=[]

def get_vectorstore_from_url(urls):
    if documents_chunks is not None or documents_chunks != "":
        documents_chunks.clear()
    for url in urls:
        loader = WebBaseLoader(url)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        documents_chunks.extend(document_chunks)

    chroma_path = 'chroma_web_llm'
    vector_store=Chroma.from_documents(documents_chunks, OpenAIEmbeddings(),persist_directory=chroma_path)
    return vector_store

def get_context_retriever_chain(vector_store):
  llm=ChatOpenAI()
  retriever = vector_store.as_retriever()
  prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user","{input}"),
      ("user","Given the above conversation, generate a serach query to look up in order to get information relevant to the conversation")
      ])

  retriever_chain = create_history_aware_retriever(llm,retriever,prompt)
  return retriever_chain

def get_conversational_rag_chain(retriver_chain):
  llm =ChatOpenAI()
  prompt = ChatPromptTemplate.from_messages([
      ("system","Answer the user's questions based on the below context : \n\n{context}"),
      MessagesPlaceholder(variable_name ="chat_history"),
      ("user","{input}")
      ])

  stuff_document_chain = create_stuff_documents_chain(llm,prompt)
  return create_retrieval_chain(retriver_chain,stuff_document_chain)

def get_response(user_query):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversational_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input" : user_query,
        })
    return response['answer']

with st.sidebar:
    st.header("Settings")
    web_site_urls = ["https://mobitel.lk/prepaid/new-connection","https://mobitel.lk/prepaid/call-management","https://mobitel.lk/how-to-recharge","https://mobitel.lk/safe-connect","https://mobitel.lk/prepaidloan","https://mobitel.lk/prepaid-ebill","https://mobitel.lk/volte-promotion"]
    
if web_site_urls is None or web_site_urls == "":
    st.info("please enter a website url")
else:
    if "chat_history" not in st.session_state:
      st.session_state.chat_history = [
          AIMessage('HI,I am a bot, How can I help you?')
      ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store =get_vectorstore_from_url(web_site_urls)
        #st.session_state.vector_store = get_vectorstore_from_pdf(pdf_paths)
    #search_results = vector_store.similarity_search('langchain', k=3)


    user_query = st.chat_input("type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(user_query))
        st.session_state.chat_history.append(AIMessage(response))

    for message in st.session_state.chat_history:
        if isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        elif isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
