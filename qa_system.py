import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from config import Config

logger = logging.getLogger(__name__)

class QASystem:
    def __init__(self, config: Config, vectorstore: Chroma):
        self.config = config
        openai_api_key = config.openai_api_key
        logger.info(f"API Key status: {'Set' if openai_api_key else 'Not set'}")
        if not openai_api_key:
            raise ValueError("OpenAI API key is not set. Please check your .env file or environment variables.")
        
        logger.info("Initializing OpenAI embeddings")
        try:
            self.embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
            logger.info("OpenAI embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OpenAI embeddings: {str(e)}")
            raise

        logger.info("Initializing ChatOpenAI model")
        try:
            self.qa_model = ChatOpenAI(api_key=openai_api_key, temperature=0)
            logger.info("ChatOpenAI model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ChatOpenAI model: {str(e)}")
            raise

        # Create prompt template
        prompt_template = """
        Answer the question based only on the following context. If the answer is not in the context, say "I don't have enough information to answer this question."

        Context:
        {context}

        Question: {question}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Create a contextual compression retriever
        base_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": config.top_k})
        compressor = LLMChainExtractor.from_llm(self.qa_model)
        retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

        # Create RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.qa_model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def answer_question(self, question: str) -> str:
        logger.info(f"Answering question: {question}")
        try:
            result = self.qa_chain({"query": question})
            logger.info(f"Raw result: {result}")  # Log the raw result
            answer = result.get('result', "No answer found")
            sources = [doc.metadata.get("source", "Unknown") for doc in result.get('source_documents', [])]
            formatted_answer = f"Answer: {answer}\nSources: {sources}"
            logger.info(f"Formatted answer: {formatted_answer}")
            return formatted_answer
        except Exception as e:
            logger.error(f"Exception occurred while answering question: {str(e)}")
            return f"An error occurred while processing the question: {str(e)}"