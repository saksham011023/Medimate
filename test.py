import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load .env
load_dotenv()

# Use a guaranteed text-generation model
HUGGINGFACE_REPO_ID = "microsoft/Phi-3-mini-4k-instruct"
# HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        # model_kwargs={"token":HF_TOKEN,
                    #   "max_length":"512"}
    )
    return llm


PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know—don't make up an answer.
Don't provide anything outside the given context.

Context:
{context}

Question:
{question}

Start the answer directly. No small talk please.
"""

def get_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

# Load FAISS
db = FAISS.load_local(
    "vectorstore/db_faiss",
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

# Build QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": get_prompt()}
)

if __name__ == "__main__":
    q = input("Write Query here: ")
    res = qa_chain.invoke({"query": q})
    print("Result:", res["result"])
    print("Source Documents:")
    for doc in res["source_documents"]:
        print("-", doc.metadata.get("source", "<unknown>"))
