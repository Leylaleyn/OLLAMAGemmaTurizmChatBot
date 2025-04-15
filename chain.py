# chain.py
# QA zincirinin oluşturulması

from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import config


def create_qa_chain(retriever):
    """
    Soru-cevap zincirini oluştur - Orijinal kodunuzdaki mantığı
    birebir takip eder
    """
    # LLM modelini başlat
    llm = Ollama(model=config.LLM_MODEL)

    # Prompt şablonunu oluştur
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=config.PROMPT_TEMPLATE,
    )

    # QA zincirini oluştur
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )

    return qa