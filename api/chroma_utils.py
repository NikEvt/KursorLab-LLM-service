"""
This module provides utilities for working with ChromaDB, including text processing,
chunking, and semantic search functionality.

Dependencies:
- chromadb: For vector database operations
- langchain: For text splitting and embeddings
- python-dotenv: For environment variable management
"""

import os
from typing import List, Dict, Optional, Tuple, Union
import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

class ChromaTextFilesManager:
    def __init__(
        self,
        client: chromadb.Client,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_function: Optional[EmbeddingFunction] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize the ChromaTextFilesManager with a ChromaDB client.
        
        Args:
            client (chromadb.Client): ChromaDB client instance
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
            embedding_function (Optional[EmbeddingFunction]): Embedding function to use.
                Should inherit from chromadb.api.types.EmbeddingFunction.
                If None, uses chromadb.utils.embedding_functions.DefaultEmbeddingFunction.
        """
        load_dotenv()
        self.client = client
        self.embedding_function = embedding_function or embedding_functions.DefaultEmbeddingFunction()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.collection_name = collection_name or "default"

    def add_text(
        self,
        text: str,
        user_id: str,
        doc_id: str,
        source_name: Optional[str] = None,
        additional_metadata: Optional[Dict] = None
    ) -> None:
        """
        Add text content to ChromaDB.
        
        Args:
            text (str): The text content to add
            user_id (str): ID of the user who owns the document
            doc_id (str): Unique identifier for the document
            source_name (Optional[str]): Name of the source file/document
            additional_metadata (Optional[Dict]): Additional metadata to store with the chunks
        """
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)

        # Prepare metadata for each chunk
        base_metadata = {
            "user_id": user_id,
            "doc_id": doc_id
        }
        
        if source_name:
            base_metadata["source"] = source_name
            
        if additional_metadata:
            base_metadata.update(additional_metadata)

        metadatas = [
            {
                **base_metadata,
                "chunk_index": str(i)
            }
            for i in range(len(chunks))
        ]

        # Create or get collection with custom embedding function
        collection_name = self.collection_name
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except:
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )

        # Add documents to collection
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))]
        )

    def search_documents(
        self,
        query: str,
        user_id: str,
        n_results: int = 5,
        doc_id: Optional[Union[str, List[str]]] = None,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for relevant documents in ChromaDB.
        
        Args:
            query (str): Search query
            user_id (str): ID of the user to search for
            n_results (int): Number of results to return
            doc_id (Optional[Union[str, List[str]]]): Specific document ID(s) to search within.
                If None, search across all documents.
                If a string, search within that specific document.
                If a list of strings, search within all specified documents.
            where (Optional[Dict]): Additional where conditions for filtering
            
        Returns:
            List[Dict]: List of search results with metadata
        """
        collection_name = self.collection_name
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except:
            return []

        # Start with the base condition
        conditions = [{"user_id": user_id}]
        
        # Add doc_id condition if specified
        if doc_id is not None:
            if isinstance(doc_id, str):
                conditions.append({"doc_id": doc_id})
            elif isinstance(doc_id, list):
                conditions.append({"doc_id": {"$in": doc_id}})
            
        # Add additional where conditions if specified
        if where:
            conditions.append(where)

        # Construct the where clause based on number of conditions
        if len(conditions) > 1:
            where_clause = {"$and": conditions}
        else:
            where_clause = conditions[0]

        # Perform search
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )

        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return formatted_results

    def delete_document(self, user_id: str, doc_id: str) -> None:
        """
        Delete a document and all its chunks from ChromaDB.
        
        Args:
            user_id (str): ID of the user who owns the document
            doc_id (str): ID of the document to delete
        """
        collection_name = self.collection_name
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            # Delete all chunks associated with the document
            collection.delete(
                where={"doc_id": doc_id}
            )
        except:
            pass

# Example usage
if __name__ == "__main__":
    # Initialize the manager
    persistent_client = chromadb.PersistentClient(path="chroma_db")
    manager = ChromaTextFilesManager(client=persistent_client, collection_name="test")

    # Example of adding text
    manager.add_text(
        text="Some example text content",
        user_id="user123",
        doc_id="doc456",
        source_name="example.txt"
    )

    # Search for relevant information
    query = "What is the main topic of the document?"
    results = manager.search_documents(query, "user123")
    for result in results:
        print(f"Content: {result['content'][:200]}...")
        print(f"Metadata: {result['metadata']}")
        print(f"Distance: {result['distance']}")
        print("---") 