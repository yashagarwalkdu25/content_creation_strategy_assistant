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

The frontend will start on [http://localhost:5173](http://localhost:5173) (or as configured).

---

## Usage
- Visit the frontend URL in your browser.
- Use the AI assistant to build your content profile and get trending ideas.
- Explore trending topics and related news in the dashboard.

---

## Notes
- Make sure all backend services are running before starting the frontend.
- For production, configure CORS and environment variables securely.
- For troubleshooting, check logs in the terminal for each service.

---

## License
MIT (or your chosen license) 