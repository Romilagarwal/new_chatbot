# Machine Troubleshooting AI Assistant

An advanced AI-powered chatbot system for industrial machine diagnostics and troubleshooting, built with **Ollama (Mistral Nemo)** and dynamic semantic search capabilities.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![Ollama](https://img.shields.io/badge/Ollama-Mistral_Nemo-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

## Features

### AI-Powered Diagnostics
- **Mistral Nemo Model**: Uses Ollama with Mistral Nemo for intelligent troubleshooting responses
- **RAG (Retrieval-Augmented Generation)**: Combines semantic search with AI generation for accurate solutions
- **Context-Aware**: Leverages historical maintenance data to provide relevant solutions
- **Dynamic Learning**: Real-time updates to the knowledge base from user contributions

### Semantic Search
- **Vector Embeddings**: Uses Sentence-BERT (all-MiniLM-L6-v2) for intelligent case matching
- **Hybrid Search**: Combines keyword matching and cosine similarity for optimal results
- **Case History**: Searches through past maintenance records to find similar issues

### Conversational Interface
- **Real-time Chat**: Natural language interaction with the AI assistant
- **Structured Diagnosis**: Form-based troubleshooting for detailed analysis
- **Multi-turn Conversations**: Maintains context throughout the discussion

### Analytics & Feedback
- **User Feedback System**: Collects ratings and comments to improve accuracy
- **Statistics Dashboard**: Tracks queries, success rates, and user contributions
- **Export Functionality**: Download chat history and correction submissions

## System Requirements

### Software
- **Ollama**: Must be installed and running locally
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python**: 3.8 or higher

## Installation

### 1. Install Ollama & Pull Model
First, install [Ollama](https://ollama.com/) on your system. Once installed, pull the required model:
```bash
ollama pull mistral-nemo
```

### 2. Clone the Repository
```bash
git clone https://github.com/Romilagarwal/new_chatbot.git
cd new_chatbot
```

### 3. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install flask flask-compress sentence-transformers pandas numpy scikit-learn python-dotenv pytz requests
```

### 5. Setup Environment Variables
Create a `.env` file in the project root:
```env
# Dataset path
DATABASE_PATH=mix_dataset_final.csv

# Ollama Host (default)
OLLAMA_HOST=http://localhost:11434
```

### 6. Prepare Dataset
Place your machine maintenance dataset CSV file (`mix_dataset_final.csv`) in the project root with the following columns:
- `Machine Type`
- `MACHINE` (Machine Name)
- `Problem Description`
- `Root Cause`
- `Action Taken`

## Usage

### Starting the Application

```bash
python app.py
```

The application will:
1. Connect to the local Ollama service
2. Create embeddings for the reference dataset (or load from memory)
3. Start the Flask server
4. Automatically open your web browser

### Using the Chatbot

#### Conversational Chat
Simply type your question in the chat interface:
```
"APMT machine showing alarm 203"
"How to fix welding spot machine not working?"
"Printer paper jam issue"
```

#### Structured Diagnosis
1. Select **Machine Type** from dropdown
2. Enter **Machine Name** (e.g., PRE-APMT, WELDING SPOT)
3. Describe the **Problem** in detail
4. Click **Diagnose Problem**

The system will:
- Find similar cases from the database
- Analyze patterns using AI
- Provide root cause analysis
- Suggest immediate actions
- Recommend preventive measures

### Contributing Solutions

Help improve the system by submitting your solved cases:
1. Click **Add Solution** button
2. Fill in the machine details and solution
3. Submit for review

## Project Structure

```text
new_chatbot/
├── app.py                      # Main Flask application
├── ollama_model.py             # Ollama API model wrapper
├── model_utils.py              # RAG and dynamic learning base
├── stats_cards_updater.py      # Analytics logic
├── templates/
│   └── index.html              # Frontend interface
├── data/                       # Generated data directory
│   ├── feedback.csv            # User feedback
│   ├── user_corrections.csv    # Submitted solutions
│   └── chat_logs.json          # Chat history
├── mix_dataset_final.csv       # Your maintenance dataset
├── .env                        # Environment configuration
└── README.md                   # This file
```

## Configuration

### Model Settings

**Mistral Nemo** configuration in `ollama_model.py`:
- **Temperature**: 0.7 (troubleshooting), 0.8 (chat)
- **Top-p**: 0.9

### Search Parameters

Adjust in `model_utils.py`:
```python
threshold = 0.60        # Similarity threshold
top_n = 3               # Number of primary results
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main application interface |
| `/health` | GET | System health check |
| `/stats` | GET | Usage statistics |
| `/chat` | POST | Conversational chat endpoint |
| `/diagnose` | POST | Structured diagnosis |
| `/feedback` | POST | Submit user feedback |
| `/submit-correction` | POST | Submit new solutions |
| `/export/chat-history` | GET | Export chat logs |
| `/export/corrections` | GET | Export user contributions |

## Troubleshooting

### Model Not Loading / Ollama API Error
Ensure Ollama is running in the background. Open a terminal and run:
```bash
ollama serve
```

### Dataset Issues
- Verify CSV format matches expected columns
- Check for missing values in critical columns
- Ensure proper encoding (UTF-8)

## Contributing

Contributions are welcome! Here's how you can help:

1. **Report Issues**: Found a bug? Open an issue
2. **Submit Solutions**: Use the built-in correction form
3. **Code Improvements**: Fork, improve, and submit PRs
4. **Documentation**: Help improve this README

## License

This project is licensed under the MIT License - see the LICENSE file for details.
