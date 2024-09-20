#Importing neccesary libraries
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tabula
import os

#Setting up our env
# from dotenv import load_dotenv
# load_dotenv()

#Getting the Hugging Face Token and GroqApiKey
os.environ['HF_TOKEN']=st.secrets["HF_TOKEN"]
groq_api_key=st.secrets["GROQ_API_KEY"]
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#Setting up streamlit
st.title("Conversational Chat Bot with PDF upload support and chat history")
st.write("Upload a PDF and chat along with the content of the PDF")
temperature = st.slider("Set your temperature as you require", 0.0, 1.0, 0.7)

# Let set the groq api key before hand since its not deployment ready yet
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192", temperature=temperature)

# Stateful management of Chat History (no session id)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Uploading the PDF
uploaded_documents = st.file_uploader("Choose a PDF file to upload", type="pdf", accept_multiple_files=True)
if uploaded_documents:
    documents = []
    for uploaded_document in uploaded_documents:
        temp_pdf = f"./temp.pdf" 
        with open(temp_pdf, 'wb') as file:
            file.write(uploaded_document.getvalue())
            file_name = uploaded_document.name
        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)

    # Creating the chunks of the document
    text_splitter3 = RecursiveCharacterTextSplitter(chunk_size=3800, chunk_overlap=1000)
    chunks3 = text_splitter3.split_documents(documents)

    # Storing final product of applying the embedder pattern to the chunks of documents and storing it in vector db
    vector_store = FAISS.from_documents(documents=chunks3, embedding=embeddings)

    # Making the vector as a retrieval class
    faiss_semantic_retriever = vector_store.as_retriever()
    bm25_retriever = BM25Retriever.from_documents(documents=chunks3)
    retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_semantic_retriever], weights=[0.5, 0.5])

    # System prompt for standalone question formulation
    context_system_prompt = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    context_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Creating a history aware retriever
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_prompt)

    # Question answering prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use four sentences maximum and keep the "
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

    # Create a chain for passing a list of Documents to a model.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Bind two chains together
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Creating storage functionality without session id
    def get_chat_history() -> BaseChatMessageHistory:
        return st.session_state.chat_history

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_chat_history()
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": "default"}
            }
        )
        st.write("Assistant:", response['answer'])
