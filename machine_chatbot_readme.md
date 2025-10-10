# Machine Troubleshooting AI Assistant

An advanced AI-powered chatbot system for industrial machine diagnostics and troubleshooting, built with **Mistral 7B Instruct v0.3** and semantic search capabilities.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-purple)

## Features

### AI-Powered Diagnostics
- **Mistral 7B Model**: Uses Mistral-7B-Instruct-v0.3 for intelligent troubleshooting responses
- **RAG (Retrieval-Augmented Generation)**: Combines semantic search with AI generation for accurate solutions
- **Context-Aware**: Leverages historical maintenance data to provide relevant solutions

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

### Production Ready
- **Optimized Performance**: FP16 precision for efficient GPU usage (~13.5GB VRAM)
- **Error Handling**: Robust fallback mechanisms for reliability
- **Scalable Architecture**: Modular design for easy maintenance and updates

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with 16GB+ VRAM (tested on RTX 4090 20GB)
- **RAM**: 16GB+ system memory
- **Storage**: 100GB+ free space (for model downloads)

### Software
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python**: 3.8 or higher
- **CUDA**: 11.7+ (for GPU acceleration)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Romilagarwal/new_chatbot.git
cd new_chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes
pip install flask flask-compress sentence-transformers
pip install pandas numpy scikit-learn python-dotenv pytz
```

### 4. Setup Environment Variables
Create a `.env` file in the project root:
```env
# Optional: HuggingFace token for gated models
HF_TOKEN=your_huggingface_token_here

# Device configuration
DEVICE=cuda

# Dataset path
DATABASE_PATH=mix_dataset_final.csv

# Cache directory (optional)
HF_HOME=D:\.cache\huggingface\hub
```

### 5. Prepare Dataset
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
1. Load the Mistral 7B model (~2-3 minutes first time)
2. Create embeddings for the reference dataset
3. Start the Flask server at `http://172.19.66.141:9352`
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

```
new_chatbot/
├── app.py                      # Main Flask application
├── advanced_model.py           # Mistral model wrapper
├── model_utils.py             # RAG and semantic search utilities
├── model_loader.py            # Model testing and verification
├── templates/
│   └── index.html             # Frontend interface
├── data/                      # Generated data directory
│   ├── feedback.csv           # User feedback
│   ├── user_corrections.csv   # Submitted solutions
│   └── chat_logs.json         # Chat history
├── mix_dataset_final.csv      # Your maintenance dataset
├── .env                       # Environment configuration
└── README.md                  # This file
```

## Configuration

### Model Settings

**Mistral 7B Instruct v0.3** configuration in `advanced_model.py`:
- **Precision**: FP16 (float16)
- **VRAM Usage**: ~13.5GB
- **Context Length**: 32,768 tokens
- **Temperature**: 0.7 (troubleshooting), 0.8 (chat)
- **Top-p**: 0.9-0.92

### Search Parameters

Adjust in `model_utils.py`:
```python
threshold = 0.60        # Similarity threshold
top_n = 3              # Number of primary results
extra_results = 15     # Additional dropdown candidates
```

### Server Settings

Modify in `app.py`:
```python
host = '172.19.66.141'  # Server IP
port = 9352             # Server port
debug = False           # Production mode
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

## Performance

### Model Performance
- **First Load**: 2-3 minutes (downloads ~87GB)
- **Subsequent Loads**: <1 minute (from cache)
- **Generation Speed**: 15-25 tokens/second
- **Latency**: Low (full GPU, no CPU offload)

### Search Performance
- **Embedding Creation**: Batched processing for efficiency
- **Query Time**: <1 second for semantic search
- **Accuracy**: Hybrid approach ensures relevant results

## Contributing

Contributions are welcome! Here's how you can help:

1. **Report Issues**: Found a bug? Open an issue
2. **Submit Solutions**: Use the built-in correction form
3. **Code Improvements**: Fork, improve, and submit PRs
4. **Documentation**: Help improve this README

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Mistral AI** for the Mistral 7B model
- **HuggingFace** for Transformers library
- **Sentence-Transformers** for embedding models
- **Flask** team for the web framework

## Contact

**Romil Agarwal**
- GitHub: [@Romilagarwal](https://github.com/Romilagarwal)
- Repository: [new_chatbot](https://github.com/Romilagarwal/new_chatbot)

## Troubleshooting

### Model Not Loading
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check disk space
# Ensure ~100GB free in cache directory
```

### Out of Memory
- Reduce `max_tokens` in generation config
- Close other GPU applications
- Consider using quantization (4-bit) if needed

### Slow Performance
- Verify CUDA is properly installed
- Check GPU utilization: `nvidia-smi`
- Ensure model is on GPU, not CPU

### Dataset Issues
- Verify CSV format matches expected columns
- Check for missing values in critical columns
- Ensure proper encoding (UTF-8)

## Future Enhancements

- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Mobile app version
- [ ] Advanced analytics dashboard
- [ ] Fine-tuning on domain-specific data
- [ ] Integration with maintenance management systems
- [ ] Real-time notification system
- [ ] Multi-user role management

---

**Built with ❤️ for industrial maintenance teams**

*Star ⭐ this repository if you find it helpful!*
