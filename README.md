# 📚 Docstóteles - IA SEMPRE ATUALIZADA (Web Scraping + RAG)

Transforme qualquer documentação em um assistente de IA atualizado!  
Crie um chat que responde sobre qualquer tecnologia, usando scraping inteligente e RAG, com ferramentas 100% gratuitas.

## ✨ O que é o Docstóteles?

O Docstóteles é uma aplicação que junta Web Scraping inteligente (Fire Crawl) com RAG (LangChain + Groq) para criar um assistente de IA que conhece qualquer documentação da web.  
Você cola o link de uma documentação (Django, React, Vue, etc), o app baixa tudo, indexa e cria um chat para perguntas e respostas super atualizadas.

## 🚀 Tecnologias Usadas

- [Streamlit](https://streamlit.io/) — Interface gráfica
- [Fire Crawl](https://firecrawl.dev/) — Web Scraping inteligente
- [Groq API](https://console.groq.com/) — LLM gratuita
- [LangChain](https://python.langchain.com/) — RAG e embeddings
- [Hugging Face](https://huggingface.co/) — Embeddings
- [FAISS](https://github.com/facebookresearch/faiss) — Vector store

## 🛠️ Instalação

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/asimov-academy/video-docstoteles-material.git
   cd video-docstoteles-material
   ```

2. **Crie e ative um ambiente virtual (recomendado):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   > No Windows, use: `.venv\Scripts\activate`

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure as chaves de API:**
   - Crie um arquivo `.env` na raiz do projeto e preencha com suas chaves:
     ```
     GROQ_API_KEY=sua_chave_groq
     FIRECRAWL_API_KEY=sua_chave_firecrawl
     FIRECRAWL_API_URL=url_firecrawl
     ```

5. **Crie as pastas necessárias:**
   ```bash
   mkdir -p data/collections
   ```

## 🏃‍♂️ Como rodar

```bash
streamlit run docstoteles/app.py
```

Acesse o app no navegador pelo link que aparecer no terminal.

---

## 📝 Como usar

### 1. Scraping

- Vá para o modo "Scraping" na barra lateral.
- Cole a URL da documentação (ex: https://docs.streamlit.io).
- Dê um nome para a coleção.
- Clique em "Iniciar Scraping".
- Aguarde o download dos arquivos.

### 2. Chat

- Selecione o modo "Chat" na barra lateral.
- Escolha a coleção que você criou.
- Pergunte qualquer coisa sobre a documentação!

---

## 🌐 Sugestões de sites para testar

- https://docs.streamlit.io
- https://python.langchain.com/docs
- https://docs.python.org/3/tutorial

---

## 📦 Estrutura do Projeto

```
docstoteles/
  app.py
  presentation/
    scraping.py
    chat.py
  service/
    scraping.py
    rag.py
data/
  collections/
requirements.txt
README.md
.env (você deve criar)
```

---

## 💡 Dicas

- O projeto é base: você pode expandir, conectar outros modelos, adicionar uploads, etc.
- Fire Crawl e Groq são gratuitos (Groq não pede cartão).
- O scraping baixa até 10 páginas por padrão (ajuste no código se quiser mais).

---

## 🧑‍💻 Contribua!

Sugestões, issues e PRs são bem-vindos!

---

# 🚦 Passo a Passo Docstóteles

## 1️⃣ Setup Básico

### 1.1 Instale as dependências

```bash
pip install streamlit python-dotenv groq firecrawl langchain langchain-community langchain-groq faiss-cpu sentence-transformers
```

### 1.2 Crie o arquivo `.env`

```env
GROQ_API_KEY=sua_chave_groq
FIRECRAWL_API_KEY=sua_chave_firecrawl
FIRECRAWL_API_URL=url_firecrawl
```

### 1.3 Estrutura do App Principal (`docstoteles/app.py`)

```python
import streamlit as st
import os
from dotenv import load_dotenv
from presentation import scraping
from presentation import chat

load_dotenv()

st.set_page_config(page_title="Docstóteles", page_icon="📚", layout="wide")
st.title("📚 Docstóteles - RAG Simples")

# Sidebar para seleção
with st.sidebar:
    st.header("Coleções")
    mode = st.radio("Modo:", ["Chat", "Scraping"])
    st.divider()
    st.subheader("Coleções Disponíveis")
    collections_dir = "data/collections"
    if os.path.exists(collections_dir):
        collections = [d for d in os.listdir(collections_dir) 
                      if os.path.isdir(os.path.join(collections_dir, d))]
        for collection in collections:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📁 {collection}")
            with col2:
                if st.button("Usar", key=f"use_{collection}"):
                    st.session_state.collection = collection
                    st.rerun()

# Estados de sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None

# Importar páginas
if mode == "Scraping":
    scraping.show()
else:
    chat.show()
```

---

## 2️⃣ Sistema de Scraping

### 2.1 Estrutura de Diretórios

```bash
mkdir -p pages data/collections
```

### 2.2 Serviço de Scraping (`docstoteles/service/scraping.py`)

```python
import os
import requests
from firecrawl import FirecrawlApp

class ScrapingService:
    def __init__(self):
        self.api_key = os.getenv("FIRECRAWL_API_KEY")
        self.api_url = os.getenv("FIRECRAWL_API_URL")
        self.app = FirecrawlApp(api_key=self.api_key, api_url=self.api_url)
    
    def scrape_website(self, url, collection_name):
        """Scraping completo em uma função"""
        try:
            # 1. Mapear URLs
            map_result = self.app.map_url(url)
            if hasattr(map_result, 'links'):
                links = map_result.links
            elif hasattr(map_result, 'data') and hasattr(map_result.data, 'links'):
                links = map_result.data.links[:10]
            else:
                links = getattr(map_result, 'links', [])[:10]
            if not links:
                raise Exception("Nenhum link encontrado!")
            print(f"Encontrados {len(links)} links")
            # 2. Fazer scraping
            scrape_result = self.app.batch_scrape_urls(links)
            # 3. Extrair dados do resultado
            if hasattr(scrape_result, 'data'):
                scraped_data = scrape_result.data
            else:
                scraped_data = scrape_result.get("data", []) if hasattr(scrape_result, 'get') else []
            # 4. Salvar arquivos
            collection_path = f"data/collections/{collection_name}"
            os.makedirs(collection_path, exist_ok=True)
            saved_count = 0
            for i, page in enumerate(scraped_data, 1):
                if hasattr(page, 'markdown') and page.markdown:
                    markdown_content = page.markdown
                elif hasattr(page, 'data') and hasattr(page.data, 'markdown'):
                    markdown_content = page.data.markdown
                elif isinstance(page, dict) and page.get("markdown"):
                    markdown_content = page["markdown"]
                else:
                    continue
                with open(f"{collection_path}/{i}.md", "w", encoding="utf-8") as f:
                    f.write(markdown_content)
                saved_count += 1
            return {"success": True, "files": saved_count}
        except Exception as e:
            print(f"Erro no scraping: {str(e)}")
            return {"success": False, "error": str(e)}
```

### 2.3 Página de Scraping (`docstoteles/presentation/scraping.py`)

```python
import streamlit as st
import os
from service.scraping import ScrapingService

def show():
    st.header("🔍 Web Scraping")
    scraper = ScrapingService()
    with st.form("scraping_form"):
        url = st.text_input("URL do site:", placeholder="https://exemplo.com")
        collection_name = st.text_input("Nome da coleção:", placeholder="minha-colecao")
        submitted = st.form_submit_button("Iniciar Scraping")
    if submitted and url and collection_name:
        with st.spinner("Extraindo conteúdo..."):
            result = scraper.scrape_website(url, collection_name)
            if result["success"]:
                st.success(f"✅ {result['files']} arquivos salvos!")
                if st.button("Ir para Chat"):
                    st.rerun()
            else:
                st.error(f"❌ Erro: {result['error']}")
    st.divider()
    st.subheader("Coleções Disponíveis")
    collections_dir = "data/collections"
    if os.path.exists(collections_dir):
        collections = [d for d in os.listdir(collections_dir) 
                      if os.path.isdir(os.path.join(collections_dir, d))]
        for collection in collections:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📁 {collection}")
```

---

## 3️⃣ Sistema RAG

### 3.1 Dependências do RAG

```bash
pip install langchain langchain-community langchain-groq faiss-cpu
```

### 3.2 Serviço RAG com LangChain (`docstoteles/service/rag.py`)

```python
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class RAGService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-8b-8192"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
        self.qa_chain = None
    def load_collection(self, collection_name):
        collection_path = f"data/collections/{collection_name}"
        loader = DirectoryLoader(
            collection_path,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()
        if not documents:
            return False
        texts = self.text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        template = """
            Use os seguintes documentos para responder a pergunta. Se você não souber a resposta, diga que não sabe.

            {context}

            Pergunta: {question}
            Resposta:
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )
        return True
    def ask_question(self, question):
        if not self.qa_chain:
            return "Nenhuma coleção carregada."
        try:
            result = self.qa_chain.run(question)
            return result
        except Exception as e:
            return f"Erro ao processar pergunta: {str(e)}"
```

### 3.3 Página de Chat (`docstoteles/presentation/chat.py`)

```python
import streamlit as st
import os
from service.scraping import ScrapingService

def show():
    st.header("🔍 Web Scraping")
    scraper = ScrapingService()
    with st.form("scraping_form"):
        url = st.text_input("URL do site:", placeholder="https://exemplo.com")
        collection_name = st.text_input("Nome da coleção:", placeholder="minha-colecao")
        submitted = st.form_submit_button("Iniciar Scraping")
    if submitted and url and collection_name:
        with st.spinner("Extraindo conteúdo..."):
            result = scraper.scrape_website(url, collection_name)
            if result["success"]:
                st.success(f"✅ {result['files']} arquivos salvos!")
                if st.button("Ir para Chat"):
                    st.rerun()
            else:
                st.error(f"❌ Erro: {result['error']}")
    st.divider()
    st.subheader("Coleções Disponíveis")
    collections_dir = "data/collections"
    if os.path.exists(collections_dir):
        collections = [d for d in os.listdir(collections_dir) 
                      if os.path.isdir(os.path.join(collections_dir, d))]
        for collection in collections:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📁 {collection}")
```

---