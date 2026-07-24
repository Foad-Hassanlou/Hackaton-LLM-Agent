"""Keyword search backends."""

from typing import List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions

from hackaton import config
from hackaton.data import ProductCatalog

COLLECTION_NAME = 'pure_keyword_search'


class KeywordSearcher:
    """Two flavours of keyword search over the product catalog:

      - normal_search: a simple, pure-Python keyword count search
      - pro_search: a ChromaDB-backed keyword search with filtering

    Each search method returns matching documents along with their category
    and original row index in the CSV.
    """

    def __init__(self, catalog: ProductCatalog, db_path=None):
        self.catalog = catalog

        # Prepare flat lists of documents and metadata
        self.documents: List[str] = []
        self.metadatas: List[dict] = []
        self.ids: List[str] = []
        for doc_id, doc_text, metadata in catalog.iter_documents():
            self.documents.append(doc_text)
            self.metadatas.append(metadata)
            self.ids.append(doc_id)

        # Initialize ChromaDB client for pro search
        self.client = chromadb.PersistentClient(path=str(db_path or config.CHROMA_DB_PATH))
        # Embeddings are not used in keyword matching, but required by API
        dummy_ef = embedding_functions.DefaultEmbeddingFunction()
        # Ensure fresh collection: delete existing data if any
        try:
            self.client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=dummy_ef
        )

        # Clear any existing documents just in case
        try:
            self.collection.delete(where={})
        except Exception:
            pass

        # Add documents to the ChromaDB collection with row_idx metadata
        self.collection.add(
            documents=self.documents,
            metadatas=self.metadatas,
            ids=self.ids
        )

    def normal_search(self, query: str, k: int = 5) -> List[Tuple[int, str, int, str]]:
        """
        Perform a simple keyword-count search over the documents.

        :param query: The search query string
        :param k: Number of top matches to return
        :return: List of tuples (score, category, row_idx, document)
        """
        query = query.strip().lower()
        results = []
        # Score each document by counting occurrences of query words
        for doc, meta in zip(self.documents, self.metadatas):
            doc_lower = doc.lower()
            score = sum(1 for word in query.split() if word in doc_lower)
            if score > 0:
                results.append((score, meta['category'], meta['row_idx'], doc))
        # Sort by descending score and return top k
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:k]

    def pro_search(self, query: str, k: int = 5,
                   category: Optional[str] = None) -> List[Tuple[str, str, int]]:
        """
        Perform a ChromaDB-backed keyword search using filter conditions.

        :param query: The search query string
        :param k: Number of top matches to return
        :param category: Optional category filter ('car', 'laptop', 'phone')
        :return: List of tuples (document, category, row_idx)
        """
        # Build filter for document text
        where_doc = {"$contains": query}
        # Prepare query args
        query_args = {
            'where_document': where_doc,
            'limit': k,
            'include': ['documents', 'metadatas']
        }
        # Add category filter if provided
        if category:
            query_args['where'] = {'category': category}

        results = self.collection.get(**query_args)
        # Include row_idx in output
        output = []
        for doc, meta in zip(results['documents'], results['metadatas']):
            output.append((doc, meta['category'], meta['row_idx']))
        return output
