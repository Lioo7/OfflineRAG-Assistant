import logging
from document_loader import DocumentLoader
from embeddings_manager import EmbeddingsManager
from qa_system import QASystem
from config import Config

logging.basicConfig(filename='app.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting the Modular LangChain-based RAG Question Answering System")
    
    try:
        logger.info("Loading configuration")
        config = Config()
        logger.info("Configuration loaded successfully")
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        print(f"Error: {str(e)}")
        return

    # Get user input for directory or file(s)
    paths = input("Enter the path to a directory or file(s), separated by commas: ").split(',')
    
    try:
        logger.info("Initializing document loader")
        doc_loader = DocumentLoader(config)
        logger.info("Loading documents")
        documents = doc_loader.load_documents(paths)
        if not documents:
            logger.error("No documents found. Exiting.")
            print("No documents found in the provided paths. Please provide a valid directory or file(s).")
            return
        logger.info(f"Loaded {len(documents)} documents")
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        print(f"Error loading documents: {str(e)}")
        return
    
    try:
        logger.info("Initializing embeddings manager")
        embeddings_manager = EmbeddingsManager(config)
        logger.info("Creating vector store")
        vectorstore = embeddings_manager.create_vectorstore(documents)
        logger.info("Vector store created successfully")
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        print(f"Error creating vector store: {str(e)}")
        return
    
    try:
        logger.info("Initializing QA system")
        qa_system = QASystem(config, vectorstore)
        logger.info("QA system initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing QA system: {str(e)}")
        print(f"Error initializing QA system: {str(e)}")
        return
    
    # Main interaction loop
    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        try:
            logger.info(f"Processing question: {question}")
            formatted_answer = qa_system.answer_question(question)
            logger.info("Answer generated successfully")
            print(formatted_answer)
            
        except Exception as e:
            logger.error(f"An error occurred while processing the question: {str(e)}")
            print(f"Sorry, an error occurred: {str(e)}")

if __name__ == "__main__":
    main()
