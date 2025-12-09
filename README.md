# 🚗 Car Sales Advisor Chatbot

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![NLP](https://img.shields.io/badge/AI-NLP-purple)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI%20%2F%20LangChain-green)

## 🤖 About The Project

**Car Sales Advisor** is an intelligent conversational chatbot that helps users find the most suitable vehicle based on their needs, preferences, and budget — using natural language instead of rigid search filters.

The chatbot understands real-life queries such as:

- “I need an SUV for a family of 4 under 700,000 TL.”
- “Show me something fuel-efficient for long commutes.”
- “I want a car with less than 80,000 km and newer than 2018.” 

The system converts structured car data into descriptive sentences and performs **semantic search** to retrieve vehicles that best match the user’s intent.

---

## ✨ Key Features

### 🧠 Natural Language Understanding
Understands free-form user input — no dropdowns, keywords, or strict filters.

### 🎯 Personalized Car Recommendations
Provides tailored suggestions based on:
- Budget  
- Model & Brand  
- Year of production  
- Engine size  
- Color  
- Mileage (km)  
- Performance or fuel efficiency  
- Preferred categories (SUV, sedan, hatchback, etc.)

### 🔍 Semantic Search
The vehicle dataset is converted into natural-language sentences and embedded into a vector space, enabling highly accurate similarity search.

### 🔄 Conversation Memory
The chatbot keeps track of the conversation, enabling multi-turn interactions, comparisons, and refinements.

### ⚙️ Retrieval-Augmented Generation (RAG)
The pipeline combines:
- **Embeddings** → semantic understanding  
- **Qdrant vector database** → similarity search  
- **LLM** → detailed, context-aware answers  

This ensures responses are grounded, relevant, and aligned with the dataset.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Core Language** | Python |
| **LLM Framework** | LangChain |
| **Embedding Model** | OpenAI / Sentence Transformers |
| **Vector Database** | Qdrant |
| **Frontend** | Streamlit |
| **API Layer** | FastAPI |
| **Data Processing** | Pandas + custom Python utilities |

---

## 📂 Project Structure

```bash
Car-Sales-Advisor/
├── api/
│   ├── main.py                     # FastAPI backend entrypoint
│
├── notebooks/
│   ├── tests.ipynb                 # Experimentation & development notebook
│
├── scripts/                        # Core business logic
│   ├── embeder.py                  # Embedding generation functions
│   ├── filters.py                  # Optional rule-based filtering
│   ├── formatter.py                # Formats responses for the chatbot
│   ├── normalize.py                # Text & data normalization utilities
│   ├── qdrant_utils.py             # Qdrant setup, inserts, and querying
│   ├── recommend.py                # Recommendation engine
│   ├── searcher.py                 # Semantic search pipeline
│   ├── test_search.py              # Search-related test cases
│   └── deneme.py                   # QDrant tests
│
├── ui/
│   ├── st_chatbot.py               # Streamlit user interface
│   └── requirements.txt            # UI dependencies
│
└── README.md                       # Project documentation
