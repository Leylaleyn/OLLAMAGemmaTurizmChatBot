# config.py
# Yapılandırma parametreleri

# Dosya yolları
PDF_PATH = "Bes_yildizli_oteller.pdf"
FAISS_INDEX_PATH = "faiss_index_oteller"

# Metin bölme parametreleri - orijinal değerler
CHUNK_SIZE = 800
CHUNK_OVERLAP = 300

# Model yapılandırmaları
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "gemma2:2b"

# Prompt şablonu
PROMPT_TEMPLATE = """
Aşağıda otellere ait bilgiler yer alıyor. Kullanıcının sorusunu bu içeriklere dayanarak yanıtla.

Eğer içerikte tam bilgi yoksa, "Bu bilgiye ulaşılamadı." diyebilirsin.

{context}

Soru: {question}
Cevap:
"""