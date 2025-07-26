# LLM Notebook RAG System

This project is a comprehensive RAG (Retrieval-Augmented Generation) system designed to answer questions about ASU, leveraging data from web scrapes, Reddit, and historical grade data. It features a Flask backend, a Next.js frontend, and uses ChromaDB for local development and Qdrant for production.

## Features

- **Multi-Source Data Ingestion**: Scrapes and processes data from ASU's website, Reddit, and CSV files of ASU grades.
- **Efficient Vector Storage**: Uses ChromaDB for fast, local vector storage and Qdrant for a scalable, production-ready cloud solution.
- **Parallelized Data Processing**: Scripts are optimized to run in parallel, significantly speeding up data ingestion and embedding.
- **RESTful API**: A Flask-based backend provides endpoints for querying the RAG system.
- **Interactive Frontend**: A Next.js application for a user-friendly interface to interact with the RAG system.

## Project Structure

```
.
├── frontend/         # Next.js frontend application
├── scripts/
│   ├── data_processing/ # Scripts for processing raw data
│   ├── deployment/      # Scripts for running the application
│   ├── scraping/        # Scripts for web scraping
│   ├── testing/         # Test scripts
│   └── vector_db/       # Scripts for building the vector database
├── src/
│   ├── rag/             # Core RAG system logic
│   ├── scrapers/        # Scraper implementations
│   └── utils/           # Utility functions and data processors
├── data/
│   ├── raw/             # Raw, unprocessed data
│   └── vector_db/       # Local ChromaDB storage
└── ...
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- An OpenAI API key

### 1. Setup the Environment

Clone the repository and set up the Python virtual environment.

```bash
git clone <repository-url>
cd new-llm-notebook
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY="your-openai-api-key"
```

### 2. Scrape the Data

To populate the RAG system, you need to scrape the data sources.

**Scrape Historical Data (run once):**

This command scrapes all historical data. It's a long-running process that should only be done once.

```bash
python scripts/scraping/historical_scrape.py
```

**Scrape Current Data:**

To get the latest data, run the smart scraper, which intelligently fetches new information.

```bash
python scripts/scraping/smart_scrape.py
```

### 3. Build the Vector Database

After scraping the data, you need to process it and store it in the ChromaDB vector store. This script processes all raw data in parallel and generates embeddings.

```bash
python scripts/vector_db/build_chroma_db.py
```

This will create a `data/vector_db` directory containing the local ChromaDB.

### 4. Run the Application

With the database built, you can now run the backend and frontend servers.

**Start the Backend API:**

```bash
python scripts/deployment/start_api_server.py
```

The API will be available at `http://localhost:3000`.

**Start the Frontend Application:**

In a separate terminal, start the Next.js frontend.

```bash
cd frontend
npm install
npm run dev
```

The frontend will be running at `http://localhost:3001`.

## Deployment

This application is configured for deployment on Render. The `render.yaml` file is set up to:

1.  Install production dependencies from `requirements-prod.txt`.
2.  Build a Qdrant vector database using the optimized `scripts/vector_db/build_rag_qdrant_fast.py` script.
3.  Start the production server using `scripts/deployment/start_production.py`.

The API server is configured to use Qdrant when deployed on Render (by detecting the `RENDER` environment variable). 