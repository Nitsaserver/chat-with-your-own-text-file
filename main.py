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

#testdata
test_data = [
    {
        "question": "What is MendZone?",
        "expected": "healthcare startup"
    },
    {
        "question": "What does MendZone do?",
        "expected": "automates reports"
    }
]

retriever = db.as_retriever(search_kwargs={"k": 8})
def get_context(query):
    docs = retriever.invoke(query)
    reranked = rerank_docs(query, docs, embeddings, top_k=3)
    return "\n".join(doc.page_content for doc in reranked)
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

def rerank_docs(query, docs, embeddings, top_k=3):
    query_embedding = embeddings.embed_query(query)
    
    scored_docs = []
    for doc in docs:
        doc_embedding = embeddings.embed_query(doc.page_content)
        
        # cosine similarity (simple version)
        score = sum(q * d for q, d in zip(query_embedding, doc_embedding))
        scored_docs.append((score, doc))
    
    # sort by score (descending)
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    return [doc for _, doc in scored_docs[:top_k]]
#testdata
test_data = [
    {
        "question": "What is the full form of RAG",
        "expected": "Retrieval Augmented Generation"
    },
    {
        "question": "What is RAG?",
        "expected": "Retrieval-Augmented Generation (RAG) is a technique used in AI systems to improve the accuracy of responses by combining information retrieval with text generation."
    }
]
def evaluate(query, answer, context, expected):
    eval_prompt = f"""
You are an evaluator.

Question: {query}
Expected Answer: {expected}
Actual Answer: {answer}
Context: {context}

Evaluate:
1. Is the answer relevant to the question? (yes/no)
2. Is the answer supported by the context? (yes/no)
3. Does the context contain the expected information? (yes/no)

Return ONLY in this format:
relevance: <yes/no>
faithfulness: <yes/no>
context_precision: <yes/no>
"""

    eval_response = llm.invoke(eval_prompt)

    return eval_response
# LCEL chain 🚀
chain = (
    {
        "context": get_context,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)
for test in test_data:
    query = test["question"]
    expected = test["expected"]

    docs = retriever.invoke(query)
    context = "\n".join(doc.page_content for doc in docs)

    answer = llm.invoke(prompt.format(context=context, question=query))

    scores = evaluate(query, answer, context, expected)

    print("\nQuestion:", query)
    print("Answer:", answer)
    print("Scores:", scores)
# Chat loop
while True:
    query = input("\nAsk: ")
    response = chain.invoke(query)
    print("\nAnswer:", response)

