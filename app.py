import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# --- Setup (runs once) ---
@st.cache_resource
def setup():
    loader = TextLoader("data.txt")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = OllamaLLM(model="llama3")

    prompt = ChatPromptTemplate.from_template("""
    Answer ONLY using the context below.
    If not found, say "I don't know".

    Context:
    {context}

    Question:
    {question}
    """)

    return retriever, llm, prompt

retriever, llm, prompt = setup()

# --- UI ---
st.title("💬 Chat with your data")

query = st.text_input("Ask something:")

if query:
    docs = retriever.invoke(query)
    context = "\n".join(doc.page_content for doc in docs)

    response = llm.invoke(prompt.format(context=context, question=query))

    st.write("### Answer:")
    st.write(response)