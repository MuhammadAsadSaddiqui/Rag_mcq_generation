
import os
import glob
import shutil
import logging
from typing import List, Dict, Any
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 50
MAX_CONTEXT_TOKENS = 3000
RETRIEVAL_K = 2


logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self, persist_directory: str = None, min_chunk_length: int = 50):

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        self.vectorstore = None
        self.retriever = None
        self.min_chunk_length = min_chunk_length

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.persist_directory = persist_directory or f"./chroma_db_{timestamp}"

    def clean_old_databases(self):
        try:
            old_dbs = glob.glob("./chroma_db_*")
            for old_db in old_dbs:
                if os.path.exists(old_db):
                    shutil.rmtree(old_db, ignore_errors=True)
                    logger.info(f"Cleaned old database at {old_db}")
        except Exception as e:
            logger.warning(f"Could not clean old databases: {e}")

    def build_index(self, documents: List[Document]) -> None:
        logger.info("Building fresh ChromaDB vector index...")

        self.clean_old_databases()

        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} text chunks")

            meaningful_chunks = []
            for chunk in chunks:
                if len(chunk.page_content.strip()) >= self.min_chunk_length:
                    meaningful_chunks.append(chunk)
                else:
                    logger.debug(f"Filtered out short chunk: {chunk.page_content[:50]}...")

            logger.info(f"Filtered to {len(meaningful_chunks)} meaningful chunks (min length: {self.min_chunk_length})")

            for i, chunk in enumerate(meaningful_chunks[:3]):
                logger.info(f"Chunk {i + 1} preview: {chunk.page_content[:200]}...")

            if not meaningful_chunks:
                raise ValueError("No meaningful content found in documents after filtering short chunks!")

            self.vectorstore = Chroma.from_documents(
                documents=meaningful_chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )

            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVAL_K}
            )

            logger.info("ChromaDB vector index built successfully")

        except Exception as e:
            logger.error(f"Error building vector index: {e}")
            raise

    def retrieve_relevant_context(self, query: str, max_tokens: int = None) -> str:

        if not self.retriever:
            raise ValueError("Vector index not built. Call build_index first.")

        max_tokens = max_tokens or MAX_CONTEXT_TOKENS

        try:
            logger.info(f"Retrieving context for query: {query}")

            # Retrieve relevant documents
            docs = self.retriever.invoke(query)

            logger.info(f"Retrieved {len(docs)} documents")

            context_parts = []
            total_length = 0

            for i, doc in enumerate(docs):
                content = doc.page_content.strip()
                logger.info(f"Doc {i + 1} preview: {content[:100]}...")

                if total_length + len(content) < max_tokens:
                    context_parts.append(content)
                    total_length += len(content)
                else:
                    # Add partial content if it fits
                    remaining = max_tokens - total_length
                    if remaining > 100:  # Only add if meaningful
                        context_parts.append(content[:remaining])
                    break

            context = "\n\n".join(context_parts)
            logger.info(f"Final context length: {len(context)} characters")

            if not context.strip():
                logger.warning("Retrieved empty context!")

            return context

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""

    def search_similar_content(self, query: str, k: int = None) -> List[Dict[str, Any]]:

        if not self.vectorstore:
            raise ValueError("Vector index not built. Call build_index first.")

        k = k or RETRIEVAL_K

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })

            logger.info(f"Found {len(formatted_results)} similar documents for query: {query}")
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching for similar content: {e}")
            return []

    def get_random_context(self, max_tokens: int = None) -> str:

        if not self.vectorstore:
            raise ValueError("Vector index not built. Call build_index first.")

        max_tokens = max_tokens or MAX_CONTEXT_TOKENS

        try:
            # Get all documents and select random ones
            import random

            # Use a broad query to get diverse results
            docs = self.retriever.invoke("content information knowledge")

            if not docs:
                logger.warning("No documents found for random context")
                return ""

            # Shuffle for randomness
            random.shuffle(docs)

            # Combine random documents
            context_parts = []
            total_length = 0

            for doc in docs:
                content = doc.page_content.strip()

                if total_length + len(content) < max_tokens:
                    context_parts.append(content)
                    total_length += len(content)
                else:
                    remaining = max_tokens - total_length
                    if remaining > 100:
                        context_parts.append(content[:remaining])
                    break

            context = "\n\n".join(context_parts)
            logger.info(f"Generated random context with length: {len(context)} characters")

            return context

        except Exception as e:
            logger.error(f"Error getting random context: {e}")
            return ""

    def get_vectorstore_stats(self) -> Dict[str, Any]:
        if not self.vectorstore:
            return {"status": "not_built"}

        try:
            collection = self.vectorstore._collection
            return {
                "status": "ready",
                "document_count": collection.count(),
                "persist_directory": self.persist_directory,
                "min_chunk_length": self.min_chunk_length
            }
        except Exception as e:
            logger.warning(f"Could not get vectorstore stats: {e}")
            return {"status": "error", "error": str(e)}

    def update_retrieval_settings(self, k: int = None, search_type: str = "similarity"):

        if not self.vectorstore:
            logger.warning("Cannot update settings - vector store not built")
            return

        search_kwargs = {"k": k or RETRIEVAL_K}

        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

        logger.info(f"Updated retrieval settings: k={search_kwargs['k']}, search_type={search_type}")

    def cleanup(self):
        try:
            if self.vectorstore:
                del self.vectorstore
            logger.info(f"Marked database for cleanup: {self.persist_directory}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")