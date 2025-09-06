# PDF_helper

[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![LlamaIndex](https://img.shields.io/badge/Framework-LlamaIndex-FF6B35?logo=llama)](https://www.llamaindex.ai/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini-8A2BE2?logo=google)](https://deepmind.google/technologies/gemini/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful, multimodal Retrieval-Augmented Generation (RAG) system that allows you to converse with your documents using **text**, **voice**, and **images**. Built to make information retrieval intuitive and seamless.

---

## ✨ Features

| Feature | Description | Technology |
| :--- | :--- | :--- |
| **📄 Text Chat** | Ask direct questions about your PDF documents. | LlamaIndex, Gemini |
| **🎤 Voice Queries** | Upload an audio clip asking a question. The app transcribes it and finds the answer. | Whisper, LlamaIndex |
| **🖼️ Image + Text** | Upload an image containing text or diagrams and ask contextual questions. | BLIP, Tesseract OCR, Gemini |
| **🧠 Smart RAG** | Combines retrieved document context with multi-modal inputs for insightful answers. | LlamaIndex, Gemini |

---

## 🛠️ Tech Stack

*   **Framework:** `LlamaIndex`
*   **LLM:** `Google Gemini 1.5 Flash`
*   **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
*   **Speech-to-Text:** `OpenAI Whisper`
*   **Image Captioning:** `Salesforce BLIP`
*   **OCR:** `Tesseract`
*   **Web UI:** `Streamlit`

---

## 📋 Sample Document

This project's knowledge base is powered by the research paper:
> **"A Scalable Machine Learning Strategy for Resource Allocation in Database"** by G. R. et al.

The RAG pipeline ingests and retrieves information from this PDF to answer your queries accurately.

---

## 🖼️ Demo & Working Model

Check out the `op_image_directory/` folder in this repository to see screenshots and GIFs of the working application!

**Sneak Peek:**
*   `app_demo_1.png`: Asking a text-based question.
*   `app_demo_2.png`: Processing a voice query.
*   `app_demo_3.gif`: Full workflow of an image-based query.

---

## 🚀 How to Run It Yourself

### 1. Prerequisites

Ensure you have Python 3.8+ installed. You will also need:
*   A Google Cloud API key with access to the Gemini API. [Get one here](https://aistudio.google.com/app/apikey).
*   Tesseract OCR installed on your system.
    *   **macOS:** `brew install tesseract`
    *   **Linux (Debian/Ubuntu):** `sudo apt install tesseract-ocr`
    *   **Windows:** [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Installation & Setup

```bash
# 1. Clone this repository
git clone <your-repo-url>
cd multimodal-gemini-assistant

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Gemini API key as an environment variable
export GOOGLE_API_KEY="your_actual_api_key_here"  # On Windows: set GOOGLE_API_KEY=your_key
