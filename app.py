# app.py

import streamlit as st
from ingest import prepare_vectorstore
from chain import create_qa_chain

# Başlık
st.title("Otel Bilgi Asistanı")

# Vektör veritabanını hazırla
@st.cache_resource
def load_qa_chain():
    vectorstore = prepare_vectorstore()
    retriever = vectorstore.as_retriever()
    qa_chain = create_qa_chain(retriever)
    return qa_chain

qa = load_qa_chain()

# Soru girişi
user_question = st.text_input("Otele dair bir soru sorun:")

# Cevaplama
if user_question:
    with st.spinner("Yanıt aranıyor..."):
        response = qa.invoke({"query": user_question})
        st.markdown("### Cevap:")
        st.write(response["result"])
