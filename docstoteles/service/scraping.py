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
            # 1. Mapear URLs - CORREÇÃO AQUI
            map_result = self.app.map_url(url)
            
            # O map_result é um objeto MapResponse, não um dict
            # Vamos acessar os links diretamente
            if hasattr(map_result, 'links'):
                links = map_result.links
            elif hasattr(map_result, 'data') and hasattr(map_result.data, 'links'):
                links = map_result.data.links[:10]
            else:
                # Se não conseguir acessar, tentar como dict (fallback)
                links = getattr(map_result, 'links', [])[:10]
            
            if not links:
                raise Exception("Nenhum link encontrado!")
            
            print(f"Encontrados {len(links)} links")
            
            # 2. Fazer scraping - CORREÇÃO: batch_scrape_urls só aceita 1 argumento
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
                # Acessar markdown do objeto page
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