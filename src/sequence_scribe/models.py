"""
Models module containing the ML models and embedders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional
import logging

# Re-export the GNN model from your existing code
from .legacy.graph_embedding_headless import SimpleGNN, UIGraphEmbedder

class ModelRegistry:
    """Registry for managing available models."""
    
    @staticmethod
    def list_available() -> Dict[str, Any]:
        """List all available models and their status."""
        models = {
            "vision_models": {
                "yolo": {
                    "description": "YOLO model for UI element detection",
                    "required_files": ["best.pt"],
                    "status": "available"
                },
                "florence2": {
                    "description": "Florence-2 model for image captioning",
                    "model_path": "Microsoft/Florence-2-base",
                    "status": "available"
                }
            },
            "embedding_models": {
                "sentence_transformers": {
                    "description": "SentenceTransformer models for text embeddings",
                    "default_model": "all-MiniLM-L6-v2",
                    "alternatives": ["all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
                    "status": "available"
                },
                "gnn": {
                    "description": "Graph Neural Network for UI element embeddings",
                    "model_class": "SimpleGNN",
                    "status": "available"
                }
            },
            "analysis_models": {
                "hmm": {
                    "description": "Hidden Markov Model for sequence analysis",
                    "library": "hmmlearn",
                    "status": "available"
                },
                "clustering": {
                    "description": "Various clustering algorithms",
                    "methods": ["kmeans", "dbscan", "hierarchical"],
                    "status": "available"
                }
            }
        }
        
        return models

class EmbeddingModel:
    """Base class for embedding models."""
    
    def __init__(self, device: str = None, logger: logging.Logger = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger(__name__)
    
    def encode(self, inputs: List[Any]) -> np.ndarray:
        """Encode inputs into embeddings."""
        raise NotImplementedError
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        raise NotImplementedError

class TextEmbeddingModel(EmbeddingModel):
    """Text embedding model using SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            try:
                self._model = SentenceTransformer(self.model_name)
                self.logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load model {self.model_name}: {e}")
                # Try fallback models
                fallback_models = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2']
                for fallback in fallback_models:
                    if fallback != self.model_name:
                        try:
                            self._model = SentenceTransformer(fallback)
                            self.logger.info(f"Loaded fallback model: {fallback}")
                            break
                        except Exception:
                            continue
                
                if self._model is None:
                    raise RuntimeError("Failed to load any SentenceTransformer model")
        
        return self._model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        return self.model.encode(texts)
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

class GraphEmbeddingModel(EmbeddingModel):
    """Graph embedding model using GNN."""
    
    def __init__(
        self, 
        hidden_dim: int = 128, 
        output_dim: int = 64, 
        num_layers: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self._embedder = None
    
    @property
    def embedder(self):
        """Lazy load the embedder."""
        if self._embedder is None:
            self._embedder = UIGraphEmbedder(
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers,
                device=self.device
            )
            self.logger.info(f"Initialized GNN embedder with output dim: {self.output_dim}")
        return self._embedder
    
    def encode(self, graphs: Dict[str, nx.Graph]) -> Dict[str, np.ndarray]:
        """Encode graphs into embeddings."""
        return self.embedder.generate_embeddings(graphs)
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.output_dim

class HybridEmbeddingModel(EmbeddingModel):
    """Hybrid model that combines text and graph embeddings."""
    
    def __init__(
        self,
        text_model_name: str = "all-MiniLM-L6-v2",
        graph_hidden_dim: int = 128,
        graph_output_dim: int = 64,
        combination_method: str = "concatenate",  # "concatenate", "average", "learned"
        **kwargs
    ):
        super().__init__(**kwargs)
        self.text_model = TextEmbeddingModel(text_model_name, **kwargs)
        self.graph_model = GraphEmbeddingModel(
            hidden_dim=graph_hidden_dim,
            output_dim=graph_output_dim,
            **kwargs
        )
        self.combination_method = combination_method
        
        if combination_method == "learned":
            # Initialize a learned combination layer
            text_dim = self.text_model.get_embedding_dimension()
            graph_dim = self.graph_model.get_embedding_dimension()
            self.combination_layer = nn.Linear(text_dim + graph_dim, max(text_dim, graph_dim))
    
    def encode(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Encode mixed data (text and graphs) into embeddings.
        
        Args:
            data: Dictionary containing 'texts' and 'graphs' keys
            
        Returns:
            Combined embeddings
        """
        text_embeddings = None
        graph_embeddings = None
        
        if 'texts' in data:
            text_embeddings = self.text_model.encode(data['texts'])
        
        if 'graphs' in data:
            graph_emb_dict = self.graph_model.encode(data['graphs'])
            # Convert to array (assuming same order as texts)
            if 'frame_order' in data:
                graph_embeddings = np.array([
                    graph_emb_dict[frame] for frame in data['frame_order']
                ])
        
        # Combine embeddings
        if text_embeddings is not None and graph_embeddings is not None:
            return self._combine_embeddings(text_embeddings, graph_embeddings)
        elif text_embeddings is not None:
            return text_embeddings
        elif graph_embeddings is not None:
            return graph_embeddings
        else:
            raise ValueError("No valid embeddings found in data")
    
    def _combine_embeddings(self, text_emb: np.ndarray, graph_emb: np.ndarray) -> np.ndarray:
        """Combine text and graph embeddings."""
        if self.combination_method == "concatenate":
            return np.concatenate([text_emb, graph_emb], axis=1)
        elif self.combination_method == "average":
            # Average after padding to same dimension
            max_dim = max(text_emb.shape[1], graph_emb.shape[1])
            if text_emb.shape[1] < max_dim:
                text_emb = np.pad(text_emb, ((0, 0), (0, max_dim - text_emb.shape[1])))
            if graph_emb.shape[1] < max_dim:
                graph_emb = np.pad(graph_emb, ((0, 0), (0, max_dim - graph_emb.shape[1])))
            return (text_emb + graph_emb) / 2
        elif self.combination_method == "learned":
            # Use learned combination
            combined = np.concatenate([text_emb, graph_emb], axis=1)
            with torch.no_grad():
                combined_tensor = torch.from_numpy(combined).float()
                result = self.combination_layer(combined_tensor)
                return result.numpy()
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of combined embeddings."""
        text_dim = self.text_model.get_embedding_dimension()
        graph_dim = self.graph_model.get_embedding_dimension()
        
        if self.combination_method == "concatenate":
            return text_dim + graph_dim
        elif self.combination_method == "average":
            return max(text_dim, graph_dim)
        elif self.combination_method == "learned":
            return max(text_dim, graph_dim)
        else:
            return text_dim

class SequenceModel:
    """Base class for sequence analysis models."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def fit(self, sequences: np.ndarray, **kwargs):
        """Fit the model to sequence data."""
        raise NotImplementedError
    
    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """Predict sequence labels or states."""
        raise NotImplementedError
    
    def score(self, sequences: np.ndarray) -> float:
        """Score the model on sequence data."""
        raise NotImplementedError

class HMMSequenceModel(SequenceModel):
    """Hidden Markov Model for sequence analysis."""
    
    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "diag",
        n_iter: int = 100,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self._model = None
    
    @property
    def model(self):
        """Lazy load the HMM model."""
        if self._model is None:
            from hmmlearn import hmm
            self._model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state
            )
            self.logger.info(f"Initialized HMM with {self.n_components} states")
        return self._model
    
    def fit(self, sequences: np.ndarray, **kwargs):
        """Fit HMM to sequence data."""
        self.model.fit(sequences)
        return self
    
    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """Predict hidden states for sequences."""
        return self.model.predict(sequences)
    
    def score(self, sequences: np.ndarray) -> float:
        """Compute log likelihood of sequences."""
        return self.model.score(sequences)
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get the learned transition matrix."""
        return self.model.transmat_
    
    def get_emission_parameters(self) -> Dict[str, np.ndarray]:
        """Get emission parameters (means and covariances)."""
        return {
            "means": self.model.means_,
            "covariances": self.model.covars_
        }

# Re-export key classes for convenience
__all__ = [
    "SimpleGNN",
    "UIGraphEmbedder", 
    "ModelRegistry",
    "EmbeddingModel",
    "TextEmbeddingModel",
    "GraphEmbeddingModel",
    "HybridEmbeddingModel",
    "SequenceModel",
    "HMMSequenceModel"
]