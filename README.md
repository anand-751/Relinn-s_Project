# Relinn's_Project

**🏗️ Project Architecture**  

User Query   <br>
   ↓
FAISS Vector Search   <br>
   ↓
Relevant Website Chunks   <br>
   ↓
Prompt + Context   <br>
   ↓
Groq LLaMA 3.1  <br>
   ↓
Answer in Console  <br>


**📂 Folder Structure**  <br>
**Relinns_Project/**   <br>
scraper.py  
preprocess.py  
embed_store.py  
chatbot.py  
├─scraped_data/  
├─vector_store/  
.env  
requirements.txt  
README.md 


<br>

🔧 **Tech Stack & Design Choices**

LLM: Groq (LLaMA 3.1 – fast inference, low latency)  
Orchestration: LangChain  
Vector DB: FAISS (fast, in-memory, CPU-friendly)  
Scraping: Requests + BeautifulSoup  
Secrets Management: .env + python-dotenv

<br>

**Why Sitemap-based Crawling?**

Avoids JavaScript-rendered navigation issues  
Ensures deep coverage of nested pages  
Production-grade crawling approach



⚙️ **Installation & Setup**     

1️⃣ Create Virtual Environment
python3 -m venv venv
source venv/bin/activate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Add Environment Variables

Create a .env file in the project root:  
GROQ_API_KEY=your_groq_api_key_here


<br>


🚀 **Running the Project** (Step-by-Step)  
**STEP 1**: Scrape Website:-(in bash)  
python scraper.py --url https://botpenguin.com/   

**STEP 2**: Preprocess & Chunk Data
python preprocess.py  
--input scraped_data/file_under_scraped_data_folder.json  

**STEP 3**: Build Embeddings & FAISS Index  
python embed_store.py \ --input scraped_data/file_under_vector_store.json  

**STEP 4**: Run Chatbot (Console)
python chatbot.py \ --index vector_store/file_under_vector_store



