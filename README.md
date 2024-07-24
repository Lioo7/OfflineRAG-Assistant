# OfflineRAG-Assistant

This system uses LangChain to implement a modular Retrieval-Augmented Generation (RAG) based question answering system. It allows users to load various file types and ask questions based on their contents.

## Features

- Modular design for easy maintenance and extensibility
- Supports multiple file formats (handled by LangChain's loaders)
- Uses Chroma for efficient storage and retrieval of document embeddings
- Provides question-answering capabilities using a pre-trained model
- Configurable parameters

## Installation

1. Clone this repository
2. Install the required packages: `pip install -r requirements.txt`

## Usage

Run the main script:

```
python main.py
```

Follow the prompts to select files or directories and ask questions.

## Configuration

You can modify the `config.py` file to change parameters such as chunk size, embedding model, and QA model.

## Note

This system is designed for ease of use, setup, and maintenance. It keeps all data local for simplicity and privacy.
