from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# LLM
llm = OllamaLLM(model="llama3")

# Load + split
loader = TextLoader("data.txt")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.split_documents(documents)

# Embeddings + DB
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 3})

# Prompt template (CLEAN 🔥)
prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")

# Format retrieved docs
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

# LCEL chain 🚀
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# Chat loop
while True:
    query = input("\nAsk: ")
    response = chain.invoke(query)
    print("\nAnswer:", response)