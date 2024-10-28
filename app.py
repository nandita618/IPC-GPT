import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constant import chroma_settings  # Ensure this exists and is correctly set up

# Model and Tokenizer Setup
checkpoint = "LaMini-T5-738M"

# Ensure that the model and tokenizer can be loaded
try:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        device_map="auto",
        torch_dtype=torch.float32
    )
except Exception as e:
    st.error(f"Error loading model: {e}")

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    # Load LLM pipeline
    llm = llm_pipeline()
    
    # Setup embedding function for Chroma
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Ensure the "db" directory exists
    if not os.path.exists("db"):
        os.makedirs("db")
    
    db = Chroma(persist_directory="db", embedding_function=embedding, client_settings=chroma_settings)
    
    # Configure retriever and Q&A chain
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    qa = qa_llm()
    generated_response = qa(instruction)
    
    # Ensure we handle missing 'result' gracefully
    answer = generated_response.get('result', "No answer found.")
    return answer, generated_response

def main():
    st.title('Search your PDF')
    
    with st.expander("About the App"):
        st.markdown(
            """
            **Generative AI-Powered Q&A App**: Enter questions related to your PDF file,
            and the app will retrieve and answer them.
            """
        )
    
    question = st.text_area("Enter your question")
    
    if st.button("Search"):
        st.info("Your Question: " + question)
        st.info("Your Answer:")
        
        answer, metadata = process_answer(question)
        st.write(answer)
        st.write("Metadata:", metadata)

if __name__ == '__main__':
    main()
