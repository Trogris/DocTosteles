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