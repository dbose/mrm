"""
RAG Retriever Module

Provides retrieval functionality for RAG (Retrieval-Augmented Generation) systems.
Currently supports FAISS-based semantic search over knowledge bases.

Example configuration:

```yaml
retriever:
  type: faiss
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  top_k: 3
  knowledge_base_path: data/knowledge_base.json
```
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FAISSRetriever:
    """FAISS-based semantic retrieval for RAG systems."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FAISS retriever.
        
        Args:
            config: Retriever configuration from model YAML
        """
        self.config = config
        self.top_k = config.get('top_k', 3)
        self.embedding_model_name = config.get(
            'embedding_model',
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Load embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        # Load knowledge base
        kb_path = Path(config.get('knowledge_base_path'))
        if not kb_path.is_absolute():
            # Resolve relative to project directory
            # Will be handled by caller
            pass
        
        with open(kb_path, 'r') as f:
            self.knowledge_base = json.load(f)
        
        logger.info(f"Loaded knowledge base with {len(self.knowledge_base)} items")
        
        # Load or create FAISS index
        self.index = self._load_or_create_index(kb_path)
    
    def _load_or_create_index(self, kb_path: Path):
        """Load existing FAISS index or create new one."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss not installed. "
                "Install with: pip install faiss-cpu"
            )
        
        # Check for existing index
        index_path = kb_path.parent / f"{kb_path.stem}.faiss"
        
        if index_path.exists():
            logger.info(f"Loading existing FAISS index: {index_path}")
            return faiss.read_index(str(index_path))
        else:
            logger.info("Creating new FAISS index...")
            # Create embeddings for knowledge base
            texts = []
            for item in self.knowledge_base:
                if isinstance(item, dict):
                    # Combine question and answer
                    text = f"{item.get('question', '')} {item.get('answer', '')}"
                else:
                    text = str(item)
                texts.append(text)
            
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=False,
                batch_size=32
            )
            
            # Create FAISS index
            import numpy as np
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)  # L2 distance (can use IP for cosine)
            index.add(np.array(embeddings).astype('float32'))
            
            # Save index for next time
            faiss.write_index(index, str(index_path))
            logger.info(f"Created and saved FAISS index: {index_path}")
            
            return index
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant documents for query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve (overrides config)
        
        Returns:
            List of retrieved documents with scores
        """
        import numpy as np
        import faiss
        
        k = top_k or self.top_k
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Build results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.knowledge_base):
                doc = self.knowledge_base[idx]
                
                # Build result dict
                if isinstance(doc, dict):
                    result = {
                        'id': doc.get('id', f'doc_{idx}'),
                        'text': f"{doc.get('question', '')} {doc.get('answer', '')}",
                        'score': float(distance),
                        'metadata': {k: v for k, v in doc.items() 
                                   if k not in ['question', 'answer', 'id']}
                    }
                else:
                    result = {
                        'id': f'doc_{idx}',
                        'text': str(doc),
                        'score': float(distance),
                        'metadata': {}
                    }
                
                results.append(result)
        
        return results
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension for this retriever."""
        return self.embedding_model.get_sentence_embedding_dimension()

