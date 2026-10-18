"""
Retrieval Service for MedQuAD Medical Q&A.
Supports:
1. Free, local TF-IDF vector retrieval (zero API keys, ultra-fast, Streamlit Cloud friendly)
2. Optional ChromaDB vector database retrieval
3. Evidence threshold scoring and source attribution
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config


class TFIDFRetriever:
    """
    Fast, local, API-key-free TF-IDF retrieval engine over MedQuAD records.
    Computes cosine similarity between user query and knowledge base corpus.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=25000,
            sublinear_tf=True
        )
        self.tfidf_matrix = None
        self.documents_df: Optional[pd.DataFrame] = None
        self.is_indexed = False

    def build_index(self, df: pd.DataFrame) -> bool:
        """
        Build the TF-IDF index from a pandas DataFrame of MedQuAD records.
        """
        if df is None or df.empty:
            self.is_indexed = False
            return False

        self.documents_df = df.copy().reset_index(drop=True)

        # Create combined searchable representation
        # Weighting: question, focus, and answer content
        search_texts = []
        for _, row in self.documents_df.iterrows():
            q = str(row.get("question", ""))
            f = str(row.get("focus", ""))
            a = str(row.get("answer", ""))
            # Repeat focus and question to give them higher relevance weighting
            combined = f"{f} {f} {q} {q} {a}"
            search_texts.append(combined)

        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(search_texts)
            self.is_indexed = True
            return True
        except Exception as e:
            self.is_indexed = False
            return False

    def retrieve(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[Dict]:
        """
        Retrieve top-k relevant documents for the given query.
        Returns a list of dicts with score and metadata.
        """
        if not self.is_indexed or self.documents_df is None or self.tfidf_matrix is None:
            return []

        query = query.strip()
        if not query:
            return []

        try:
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        except Exception:
            return []

        # Get top-k indices sorted descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            row = self.documents_df.iloc[idx]
            results.append({
                "id": str(row.get("id", "")),
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer", "")),
                "focus": str(row.get("focus", "")),
                "qtype": str(row.get("qtype", "")),
                "source": str(row.get("source", "NIH MedQuAD")),
                "url": str(row.get("url", "")),
                "score": round(score, 4),
                "relevance_pct": f"{min(round(score * 100, 1), 100.0)}%"
            })

        return results


class ChromaRetriever:
    """
    ChromaDB-backed semantic vector retrieval.
    Initialized optionally if chromadb is available and populated.
    """

    def __init__(self, persist_dir: Path = config.CHROMA_DIR):
        self.persist_dir = persist_dir
        self.client = None
        self.collection = None
        self.is_ready = False
        self._init_client()

    def _init_client(self):
        try:
            import chromadb
            from chromadb.config import Settings
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            count = self.collection.count()
            self.is_ready = (count > 0)
        except Exception:
            self.is_ready = False

    def index_documents(self, df: pd.DataFrame) -> int:
        """Add documents to ChromaDB collection."""
        if self.collection is None or df.empty:
            return 0
        try:
            ids = [str(row["id"]) for _, row in df.iterrows()]
            documents = [f"Question: {row['question']}\nFocus: {row.get('focus', '')}\nAnswer: {row['answer']}" for _, row in df.iterrows()]
            metadatas = [
                {
                    "question": str(row.get("question", "")),
                    "focus": str(row.get("focus", "")),
                    "qtype": str(row.get("qtype", "")),
                    "source": str(row.get("source", "")),
                    "url": str(row.get("url", ""))
                }
                for _, row in df.iterrows()
            ]
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            self.is_ready = True
            return self.collection.count()
        except Exception:
            return 0

    def retrieve(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> List[Dict]:
        if not self.is_ready or self.collection is None:
            return []
        try:
            results = self.collection.query(query_texts=[query], n_results=top_k)
            retrieved = []
            if results and "documents" in results and results["documents"]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
                ids = results["ids"][0] if "ids" in results else [""] * len(docs)
                distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

                for doc, meta, doc_id, dist in zip(docs, metas, ids, distances):
                    # Chroma cosine distance = 1 - cosine_similarity
                    sim = max(0.0, 1.0 - dist)
                    retrieved.append({
                        "id": doc_id,
                        "question": meta.get("question", ""),
                        "answer": doc.split("Answer: ")[-1] if "Answer: " in doc else doc,
                        "focus": meta.get("focus", ""),
                        "qtype": meta.get("qtype", ""),
                        "source": meta.get("source", "NIH MedQuAD"),
                        "url": meta.get("url", ""),
                        "score": round(sim, 4),
                        "relevance_pct": f"{min(round(sim * 100, 1), 100.0)}%"
                    })
            return retrieved
        except Exception:
            return []


class RetrievalService:
    """
    Unified retrieval service managing local TF-IDF and optional ChromaDB.
    Guarantees reliable, API-key-free answering.
    """

    def __init__(self):
        self.tfidf_retriever = TFIDFRetriever()
        self.chroma_retriever = ChromaRetriever()
        self.df_data: Optional[pd.DataFrame] = None
        self.load_data()

    def load_data(self) -> bool:
        """
        Load records from processed CSV or fallback sample CSV.
        """
        target_path = None
        if config.PROCESSED_CSV_PATH.exists() and config.PROCESSED_CSV_PATH.stat().st_size > 0:
            target_path = config.PROCESSED_CSV_PATH
        elif config.SAMPLE_CSV_PATH.exists() and config.SAMPLE_CSV_PATH.stat().st_size > 0:
            target_path = config.SAMPLE_CSV_PATH

        if target_path is not None:
            try:
                self.df_data = pd.read_csv(target_path)
                return self.tfidf_retriever.build_index(self.df_data)
            except Exception:
                pass
        return False

    def search(self, query: str, top_k: int = config.DEFAULT_TOP_K) -> Tuple[List[Dict], bool]:
        """
        Search knowledge base using local TF-IDF retriever.
        Returns:
            results: List of retrieved evidence document dictionaries
            has_sufficient_evidence: True if top match score >= SIMILARITY_THRESHOLD
        """
        if not self.tfidf_retriever.is_indexed:
            self.load_data()

        results = self.tfidf_retriever.retrieve(query, top_k=top_k)
        if not results:
            return [], False

        top_score = results[0]["score"]
        has_sufficient_evidence = (top_score >= config.SIMILARITY_THRESHOLD)
        return results, has_sufficient_evidence

    def get_dataset_stats(self) -> Dict:
        """Return summary statistics of loaded knowledge base."""
        if self.df_data is None or self.df_data.empty:
            return {
                "total_records": 0,
                "sources_count": 0,
                "unique_topics": 0,
                "is_indexed": False
            }
        return {
            "total_records": len(self.df_data),
            "sources_count": int(self.df_data["source"].nunique()) if "source" in self.df_data else 0,
            "unique_topics": int(self.df_data["focus"].nunique()) if "focus" in self.df_data else 0,
            "is_indexed": self.tfidf_retriever.is_indexed
        }


# Global singleton instance
retrieval_service = RetrievalService()
