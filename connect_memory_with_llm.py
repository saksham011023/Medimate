import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load .env
load_dotenv()


# HUGGINGFACE_REPO_ID = "google/flan-t5-small"

def load_llm():
    llm = HuggingFacePipeline.from_model_id(

        
        model_id="google/flan-t5-small",
        task="text2text-generation",
        model_kwargs={"temperature": 0.7, "max_length": 512}
        

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

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
