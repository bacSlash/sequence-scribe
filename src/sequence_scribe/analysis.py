"""
Analysis module that wraps HMM and other sequence analysis functionality.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np

from .legacy import graph_hmm_headless
from .legacy import naive_action_analysis

class HMMAnalyzer:
    """
    Analyzes embeddings using Hidden Markov Models.
    
    This wraps the functionality from your HMM analysis modules.
    """
    
    def __init__(self, output_dir: str = None, logger: logging.Logger = None):
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze(
        self,
        embeddings_file: str,
        output_dir: str = None,
        n_states: int = 3,
        covariance_type: str = "diag",
        n_iter: int = 100,
        train_ratio: float = 2/3,
        pca_components: int = 20,
        normalize: bool = True,
        apply_standard_scaling: bool = True,
        detect_outliers: bool = True,
        z_threshold: float = 3.0,
        embedding_type: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze embeddings using HMM.
        
        Args:
            embeddings_file: Path to embeddings NPZ file
            output_dir: Output directory for results
            n_states: Number of HMM states
            covariance_type: HMM covariance type
            n_iter: Maximum iterations for training
            train_ratio: Ratio of data for training
            pca_components: Number of PCA components
            normalize: Whether to normalize embeddings
            apply_standard_scaling: Whether to apply standard scaling
            detect_outliers: Whether to detect and handle outliers
            z_threshold: Z-score threshold for outlier detection
            embedding_type: Type of embeddings ("gnn", "text", "auto")
            **kwargs: Additional parameters
            
        Returns:
            Analysis results dictionary
        """
        if output_dir is None:
            output_dir = Path(self.output_dir) / f"hmm_analysis_{Path(embeddings_file).stem}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Analyzing embeddings: {embeddings_file}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"HMM parameters: states={n_states}, covariance={covariance_type}")
        
        try:
            # Auto-detect embedding type if needed
            if embedding_type == "auto":
                embedding_type = self._detect_embedding_type(embeddings_file)
                self.logger.info(f"Auto-detected embedding type: {embedding_type}")
            
            # Choose analysis method based on embedding type
            if embedding_type == "gnn":
                results = self._analyze_gnn_embeddings(
                    embedding_file=embeddings_file,
                    output_dir=str(output_dir),
                    n_states=n_states,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    train_ratio=train_ratio,
                    pca_components=pca_components,
                    normalize=normalize,
                    apply_standard_scaling=apply_standard_scaling,
                    detect_and_handle_outliers=detect_outliers,
                    z_threshold=z_threshold,
                    **kwargs
                )
            else:  # text embeddings
                results = self._analyze_text_embeddings(
                    embeddings_file=embeddings_file,
                    output_dir=str(output_dir),
                    n_states=n_states,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                    train_ratio=train_ratio,
                    pca_components=pca_components,
                    normalize=normalize,
                    apply_standard_scaling=apply_standard_scaling,
                    detect_outliers_flag=detect_outliers,
                    z_threshold=z_threshold,
                    **kwargs
                )
            
            # Add metadata
            results["analysis_metadata"] = {
                "embeddings_file": embeddings_file,
                "embedding_type": embedding_type,
                "output_directory": str(output_dir),
                "parameters": {
                    "n_states": n_states,
                    "covariance_type": covariance_type,
                    "n_iter": n_iter,
                    "train_ratio": train_ratio,
                    "pca_components": pca_components,
                    "normalize": normalize,
                    "apply_standard_scaling": apply_standard_scaling,
                    "detect_outliers": detect_outliers,
                    "z_threshold": z_threshold
                }
            }
            
            self.logger.info("HMM analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"HMM analysis failed: {e}")
            raise
    
    def _detect_embedding_type(self, embeddings_file: str) -> str:
        """
        Auto-detect the type of embeddings based on file structure.
        
        Args:
            embeddings_file: Path to embeddings file
            
        Returns:
            Detected embedding type ("gnn" or "text")
        """
        try:
            data = np.load(embeddings_file, allow_pickle=True)
            
            # Check for GNN-specific structure
            if len(data.files) > 10 and all(isinstance(key, str) and 'frame_' in key for key in data.files[:5]):
                return "gnn"
            
            # Check for text embedding structure
            if 'embeddings' in data.files or 'observations' in data.files:
                return "text"
            
            # Default fallback
            return "text"
            
        except Exception as e:
            self.logger.warning(f"Could not auto-detect embedding type: {e}")
            return "text"
    
    def _analyze_gnn_embeddings(self, **kwargs) -> Dict[str, Any]:
        """Analyze GNN embeddings using graph HMM module."""
        try:
            result = graph_hmm_headless.process_gnn_embeddings(**kwargs)
            return {
                "status": "success",
                "embedding_type": "gnn",
                "train_log_likelihood": result.get("train_log_likelihood"),
                "test_log_likelihood": result.get("test_log_likelihood"),
                "n_states": result.get("n_states"),
                "pca_components": result.get("pca_components")
            }
        except Exception as e:
            self.logger.error(f"GNN embedding analysis failed: {e}")
            raise
    
    def _analyze_text_embeddings(self, **kwargs) -> Dict[str, Any]:
        """Analyze text embeddings using naive analysis module."""
        try:
            result = naive_action_analysis.process_embeddings_analysis(**kwargs)
            return {
                "status": "success",
                "embedding_type": "text",
                "train_log_likelihood": result.get("train_log_likelihood"),
                "test_log_likelihood": result.get("test_log_likelihood"),
                "n_states": result.get("n_states"),
                "pca_components": result.get("pca_components"),
                "output_directory": result.get("output_directory")
            }
        except Exception as e:
            self.logger.error(f"Text embedding analysis failed: {e}")
            raise
    
    def compare_embeddings(
        self,
        embeddings_files: Dict[str, str],
        output_dir: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare different embedding types using HMM analysis.
        
        Args:
            embeddings_files: Dictionary mapping method names to embedding file paths
            output_dir: Output directory
            **kwargs: Additional parameters for HMM analysis
            
        Returns:
            Comparison results
        """
        if output_dir is None:
            output_dir = Path(self.output_dir) / "embedding_comparison"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Comparing embeddings: {list(embeddings_files.keys())}")
        
        comparison_results = {
            "embeddings_compared": list(embeddings_files.keys()),
            "results_by_method": {},
            "comparison_metrics": {}
        }
        
        # Analyze each embedding type
        for method_name, embeddings_file in embeddings_files.items():
            try:
                method_output_dir = output_dir / method_name
                result = self.analyze(
                    embeddings_file=embeddings_file,
                    output_dir=str(method_output_dir),
                    **kwargs
                )
                comparison_results["results_by_method"][method_name] = result
                
            except Exception as e:
                self.logger.error(f"Failed to analyze {method_name} embeddings: {e}")
                comparison_results["results_by_method"][method_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Compute comparison metrics
        successful_results = {
            k: v for k, v in comparison_results["results_by_method"].items()
            if v.get("status") == "success"
        }
        
        if len(successful_results) >= 2:
            comparison_results["comparison_metrics"] = self._compute_comparison_metrics(
                successful_results
            )
        
        # Save comparison results
        comparison_file = output_dir / "comparison_results.json"
        with open(comparison_file, 'w') as f:
            # Make results JSON serializable
            json_results = self._make_json_serializable(comparison_results)
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Comparison completed. Results saved to: {comparison_file}")
        return comparison_results
    
    def _compute_comparison_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute comparison metrics between different embedding methods."""
        metrics = {}
        
        # Extract log likelihoods
        train_lls = {}
        test_lls = {}
        
        for method, result in results.items():
            if result.get("train_log_likelihood") is not None:
                train_lls[method] = result["train_log_likelihood"]
            if result.get("test_log_likelihood") is not None:
                test_lls[method] = result["test_log_likelihood"]
        
        # Compare log likelihoods
        if len(train_lls) >= 2:
            best_train_method = max(train_lls, key=train_lls.get)
            metrics["best_train_method"] = best_train_method
            metrics["train_log_likelihoods"] = train_lls
        
        if len(test_lls) >= 2:
            best_test_method = max(test_lls, key=test_lls.get)
            metrics["best_test_method"] = best_test_method
            metrics["test_log_likelihoods"] = test_lls
        
        # Compute differences
        if len(test_lls) == 2:
            methods = list(test_lls.keys())
            ll_diff = test_lls[methods[0]] - test_lls[methods[1]]
            metrics["test_ll_difference"] = {
                f"{methods[0]}_vs_{methods[1]}": ll_diff
            }
        
        return metrics
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types and other non-serializable objects to JSON-compatible types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


class ClusteringAnalyzer:
    """
    Alternative analyzer using clustering methods instead of HMM.
    """
    
    def __init__(self, output_dir: str = None, logger: logging.Logger = None):
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze(
        self,
        embeddings_file: str,
        method: str = "kmeans",
        n_clusters: int = 3,
        output_dir: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze embeddings using clustering methods.
        
        Args:
            embeddings_file: Path to embeddings file
            method: Clustering method ("kmeans", "dbscan", "hierarchical")
            n_clusters: Number of clusters (for methods that require it)
            output_dir: Output directory
            **kwargs: Additional parameters
            
        Returns:
            Clustering results
        """
        if output_dir is None:
            output_dir = Path(self.output_dir) / f"clustering_analysis_{Path(embeddings_file).stem}"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Clustering analysis with {method} on: {embeddings_file}")
        
        try:
            # Load embeddings
            data = np.load(embeddings_file, allow_pickle=True)
            
            if 'embeddings' in data:
                embeddings = data['embeddings']
                frame_names = data.get('frame_names', [f"frame_{i}" for i in range(len(embeddings))])
            else:
                # Handle GNN embeddings format
                embeddings_dict = {key: data[key] for key in data.files}
                frame_names = list(embeddings_dict.keys())
                embeddings = np.array([embeddings_dict[name] for name in frame_names])
            
            # Apply clustering
            if method == "kmeans":
                from sklearn.cluster import KMeans
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
            elif method == "dbscan":
                from sklearn.cluster import DBSCAN
                clusterer = DBSCAN(**kwargs)
            elif method == "hierarchical":
                from sklearn.cluster import AgglomerativeClustering
                clusterer = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
            else:
                raise ValueError(f"Unknown clustering method: {method}")
            
            # Fit clustering
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # Create visualizations
            self._create_clustering_visualizations(
                embeddings, cluster_labels, frame_names, method, output_dir
            )
            
            # Compute metrics
            metrics = self._compute_clustering_metrics(embeddings, cluster_labels)
            
            results = {
                "status": "success",
                "method": method,
                "n_clusters": len(np.unique(cluster_labels)),
                "cluster_labels": cluster_labels.tolist(),
                "frame_names": frame_names,
                "metrics": metrics,
                "output_directory": str(output_dir)
            }
            
            # Save results
            results_file = output_dir / "clustering_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Clustering analysis completed: {len(np.unique(cluster_labels))} clusters found")
            return results
            
        except Exception as e:
            self.logger.error(f"Clustering analysis failed: {e}")
            raise
    
    def _create_clustering_visualizations(
        self, 
        embeddings: np.ndarray, 
        cluster_labels: np.ndarray, 
        frame_names: List[str],
        method: str,
        output_dir: Path
    ):
        """Create visualization plots for clustering results."""
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            
            # Apply PCA for visualization
            if embeddings.shape[1] > 2:
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(embeddings)
            else:
                embeddings_2d = embeddings
            
            # Create scatter plot
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(
                embeddings_2d[:, 0], 
                embeddings_2d[:, 1], 
                c=cluster_labels, 
                cmap='tab10', 
                alpha=0.7
            )
            plt.colorbar(scatter, label='Cluster')
            plt.title(f'{method.title()} Clustering Results')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            
            # Add frame annotations
            for i, frame_name in enumerate(frame_names):
                if i % 5 == 0:  # Annotate every 5th frame to avoid clutter
                    plt.annotate(
                        frame_name.split('_')[-1].split('.')[0] if '_' in frame_name else str(i),
                        (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        fontsize=8, alpha=0.7
                    )
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{method}_clustering.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create clustering visualization: {e}")
    
    def _compute_clustering_metrics(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> Dict:
        """Compute clustering quality metrics."""
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            
            metrics = {}
            
            if len(np.unique(cluster_labels)) > 1:
                metrics["silhouette_score"] = float(silhouette_score(embeddings, cluster_labels))
                metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(embeddings, cluster_labels))
            
            # Cluster sizes
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            metrics["cluster_sizes"] = {int(label): int(count) for label, count in zip(unique_labels, counts)}
            metrics["n_clusters"] = len(unique_labels)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to compute clustering metrics: {e}")
            return {}
