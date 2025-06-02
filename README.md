# Content Creation Strategy Assistant

## Project Overview
This project is an AI-powered assistant for content creators. It helps you discover trending topics, analyze trends, and develop a winning content strategy using a conversational interface and real-time data streams.

---

## Backend Setup

### 1. Prerequisites
- **Python 3.9+** (Recommended)
- **pip** (Python package manager)

### 2. Clone the Repository
```bash
git clone <your-repo-url>
cd content_creation_strategy_assistant
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory with your API keys and configuration. Example:
```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env
PINECONE_INDEX_NAME=trending-topics-phase1
```

### 5. Run Backend Services

#### a. Data Ingestion Service
```bash
python data_ingestion_service.py
```

#### b. Trending Topics API
```bash
python trending_topic_api.py
```

#### c. Chatbot API
```bash
python bot.py
```

Each service will start on its configured port (see code or .env for details).

---

## Frontend Setup

### 1. Prerequisites
- **Node.js** (v16+ recommended)
- **npm** (Node package manager)

### 2. Install Frontend Dependencies
```bash
cd Frontend
npm install
```

### 3. Start the Frontend Development Server
```bash
npm run dev
```

## Screenshots
---
### 1. UI for trending topics 
<img width="1512" alt="Screenshot 2025-06-02 at 5 50 43 PM" src="https://github.com/user-attachments/assets/e7448789-6ad5-4e12-9d97-eacce666454e" />

### 2. Chat bot (build a smart profile)

<img width="1454" alt="Screenshot 2025-06-02 at 5 59 05 PM" src="https://github.com/user-attachments/assets/1c2c06e8-7470-4513-9b15-53e2570418f6" />

### 3. Chat bot (content generation and feebback)

<img width="1429" alt="Screenshot 2025-06-02 at 6 01 23 PM" src="https://github.com/user-attachments/assets/062c9d9e-de65-439f-ab1c-0e1de464a3ec" />

<img width="1447" alt="Screenshot 2025-06-02 at 6 01 33 PM" src="https://github.com/user-attachments/assets/d01c43db-182d-40c2-8333-1f995daa9874" />

---
