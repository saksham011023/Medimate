import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA 
from  langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from dotenv import load_dotenv


load_dotenv()

from dotenv import load_dotenv
DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=["context","question"])
    return prompt

def load_llm():
    llm = HuggingFacePipeline.from_model_id(

        
        model_id="google/flan-t5-small",
        task="text2text-generation",
        model_kwargs={"temperature": 0.7, "max_length": 512}
        

    )

    return llm

def main():
    st.title("Ask Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages=[]
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user','content':prompt})

        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer the user's question.
            If you don't know the answer, just say that you don't know—don't make up an answer.
            Don't provide anything outside the given context.

            Context:
            {context}

            Question:
            {question}

            Start the answer directly. No small talk please.
            """
    

        try:
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store.")

            
        # Build QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            response=qa_chain.invoke({'query':prompt})

            results=response["result"]
            source_documents=response["source_documents"]
            final_result=results+"\n\nSource Docs:\n"+str(source_documents)
            # response="Hi, I am Medibot!"
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role':'assistant','content':final_result})

        except Exception as e:
            st.error(f"Error : {str(e)}")


if __name__=="__main__":
    main()