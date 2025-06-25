"""
Core SequenceScribe class that orchestrates the entire pipeline.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings

from .processors import ActionProcessor, GNNProcessor, NaiveProcessor
from .analysis import HMMAnalyzer
from .utils import setup_logging, validate_file_path, create_output_dir

class SequenceScribe:
    """
    Main class for UI sequence analysis.
    
    This class provides a unified interface for processing videos,
    extracting UI elements, creating embeddings, and analyzing sequences.
    
    Example:
        >>> scribe = SequenceScribe(output_dir="./results")
        >>> results = scribe.process_video("demo.mp4", method="gnn")
        >>> analysis = scribe.analyze_sequence(results['embeddings'])
    """
    
    def __init__(
        self,
        output_dir: str = None,
        models_dir: str = None,
        log_level: str = "INFO",
        device: str = None
    ):
        """
        Initialize SequenceScribe.
        
        Args:
            output_dir: Directory for outputs (default: ./sequence_scribe_output)
            models_dir: Directory containing model weights
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        # Setup directories
        self.output_dir = create_output_dir(output_dir or "./sequence_scribe_output")
        self.models_dir = models_dir or self._find_models_dir()
        
        # Setup logging
        self.logger = setup_logging(log_level, self.output_dir)
        
        # Device setup
        import torch
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"Initialized SequenceScribe with device: {self.device}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Initialize processors (lazy loading)
        self._action_processor = None
        self._gnn_processor = None
        self._naive_processor = None
        self._hmm_analyzer = None
    
    def _find_models_dir(self) -> Optional[str]:
        """Find models directory in package or current directory."""
        # Try package location first
        package_dir = Path(__file__).parent
        models_dir = package_dir / "weights"
        if models_dir.exists():
            return str(models_dir)
            
        # Try current directory
        current_models = Path("./src/weights")
        if current_models.exists():
            return str(current_models)
            
        self.logger.warning("Models directory not found. Some features may not work.")
        return None
    
    @property
    def action_processor(self) -> ActionProcessor:
        """Lazy-loaded action processor."""
        if self._action_processor is None:
            self._action_processor = ActionProcessor(
                models_dir=self.models_dir,
                device=self.device,
                logger=self.logger
            )
        return self._action_processor
    
    @property
    def gnn_processor(self) -> GNNProcessor:
        """Lazy-loaded GNN processor."""
        if self._gnn_processor is None:
            self._gnn_processor = GNNProcessor(
                output_dir=self.output_dir,
                device=self.device,
                logger=self.logger
            )
        return self._gnn_processor
    
    @property
    def naive_processor(self) -> NaiveProcessor:
        """Lazy-loaded naive processor."""
        if self._naive_processor is None:
            self._naive_processor = NaiveProcessor(
                output_dir=self.output_dir,
                logger=self.logger
            )
        return self._naive_processor
    
    @property
    def hmm_analyzer(self) -> HMMAnalyzer:
        """Lazy-loaded HMM analyzer."""
        if self._hmm_analyzer is None:
            self._hmm_analyzer = HMMAnalyzer(
                output_dir=self.output_dir,
                logger=self.logger
            )
        return self._hmm_analyzer
    
    def process_video(
        self,
        video_path: str,
        method: str = "auto",
        output_subdir: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a video to extract UI action sequences.
        
        Args:
            video_path: Path to video file
            method: Processing method ('auto', 'gnn', 'naive', 'both')
            output_subdir: Subdirectory name for this analysis
            **kwargs: Additional parameters for processors
            
        Returns:
            Dictionary containing:
            - elements: List of extracted UI elements
            - embeddings_path: Path to embeddings file
            - csv_path: Path to elements CSV
            - method_used: Method that was used
            - analysis_results: Results from analysis
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If method is invalid
        """
        # Validate inputs
        video_path = validate_file_path(video_path)
        if method not in ["auto", "gnn", "naive", "both"]:
            raise ValueError(f"Invalid method: {method}. Must be one of: auto, gnn, naive, both")
        
        # Create output directory for this video
        video_name = Path(video_path).stem
        if output_subdir:
            analysis_dir = Path(self.output_dir) / output_subdir
        else:
            analysis_dir = Path(self.output_dir) / f"{video_name}_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Method: {method}")
        self.logger.info(f"Output directory: {analysis_dir}")
        
        # Step 1: Extract UI elements from video
        self.logger.info("Step 1: Extracting UI elements from video frames...")
        elements, csv_path = self.action_processor.extract_sequence_from_video(
            video_path, str(analysis_dir), **kwargs
        )
        
        if not elements:
            raise RuntimeError("No UI elements were extracted from the video")
            
        self.logger.info(f"Extracted {len(elements)} UI elements")
        
        # Step 2: Create embeddings based on method
        embeddings_results = {}
        
        if method == "auto":
            # Auto-select based on data characteristics
            method = self._auto_select_method(elements)
            self.logger.info(f"Auto-selected method: {method}")
        
        if method in ["gnn", "both"]:
            self.logger.info("Step 2a: Creating graph embeddings...")
            try:
                gnn_results = self.gnn_processor.create_embeddings(
                    csv_path, output_dir=str(analysis_dir / "gnn"), **kwargs
                )
                embeddings_results["gnn"] = gnn_results
                self.logger.info("GNN embeddings created successfully")
            except Exception as e:
                self.logger.error(f"GNN processing failed: {e}")
                if method == "gnn":
                    raise
        
        if method in ["naive", "both"]:
            self.logger.info("Step 2b: Creating text embeddings...")
            try:
                naive_results = self.naive_processor.create_embeddings(
                    csv_path, output_dir=str(analysis_dir / "naive"), **kwargs
                )
                embeddings_results["naive"] = naive_results
                self.logger.info("Text embeddings created successfully")
            except Exception as e:
                self.logger.error(f"Naive processing failed: {e}")
                if method == "naive":
                    raise
        
        # Step 3: Analyze sequences (optional, based on kwargs)
        analysis_results = {}
        if kwargs.get("analyze_sequences", True):
            self.logger.info("Step 3: Analyzing sequences...")
            for emb_method, emb_results in embeddings_results.items():
                if "embeddings_file" in emb_results:
                    try:
                        analysis = self.hmm_analyzer.analyze(
                            emb_results["embeddings_file"],
                            output_dir=str(analysis_dir / f"{emb_method}_analysis"),
                            **kwargs
                        )
                        analysis_results[emb_method] = analysis
                        self.logger.info(f"Completed {emb_method} sequence analysis")
                    except Exception as e:
                        self.logger.error(f"Analysis failed for {emb_method}: {e}")
        
        # Compile results
        results = {
            "video_path": video_path,
            "elements": elements,
            "csv_path": csv_path,
            "method_used": method,
            "embeddings": embeddings_results,
            "analysis": analysis_results,
            "output_directory": str(analysis_dir),
            "summary": {
                "num_elements": len(elements),
                "methods_used": list(embeddings_results.keys()),
                "analyses_completed": list(analysis_results.keys())
            }
        }
        
        # Save summary
        self._save_summary(results, analysis_dir / "summary.json")
        
        self.logger.info("Video processing completed successfully!")
        return results
    
    def _auto_select_method(self, elements: List[Dict]) -> str:
        """
        Automatically select the best method based on data characteristics.
        
        Args:
            elements: List of UI elements
            
        Returns:
            Selected method name
        """
        # Simple heuristics for method selection
        num_elements = len(elements)
        
        # Count interactive elements
        interactive_count = sum(1 for elem in elements if elem.get("Interactivity", False))
        interactive_ratio = interactive_count / num_elements if num_elements > 0 else 0
        
        # Count elements with content
        content_count = sum(1 for elem in elements if elem.get("Content", "").strip())
        content_ratio = content_count / num_elements if num_elements > 0 else 0
        
        # Decision logic
        if num_elements < 50:
            # Small datasets work well with naive approach
            return "naive"
        elif interactive_ratio > 0.3 and content_ratio < 0.6:
            # Many interactive elements but little text content -> GNN
            return "gnn"
        elif content_ratio > 0.7:
            # Lots of text content -> naive approach
            return "naive"
        else:
            # Mixed or uncertain -> use both
            return "both"
    
    def analyze_sequence(
        self,
        embeddings_file: str,
        method: str = "hmm",
        output_dir: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze pre-computed embeddings.
        
        Args:
            embeddings_file: Path to embeddings file
            method: Analysis method
            output_dir: Output directory
            **kwargs: Additional parameters
            
        Returns:
            Analysis results
        """
        embeddings_file = validate_file_path(embeddings_file)
        
        if output_dir is None:
            output_dir = Path(self.output_dir) / f"analysis_{Path(embeddings_file).stem}"
        
        self.logger.info(f"Analyzing embeddings: {embeddings_file}")
        
        if method == "hmm":
            return self.hmm_analyzer.analyze(embeddings_file, output_dir, **kwargs)
        else:
            raise ValueError(f"Unknown analysis method: {method}")
    
    def compare_methods(
        self,
        video_path: str,
        methods: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare different processing methods on the same video.
        
        Args:
            video_path: Path to video file
            methods: List of methods to compare (default: ['gnn', 'naive'])
            **kwargs: Additional parameters
            
        Returns:
            Comparison results
        """
        if methods is None:
            methods = ['gnn', 'naive']
        
        self.logger.info(f"Comparing methods {methods} on video: {video_path}")
        
        # Process with multiple methods
        results = self.process_video(
            video_path, 
            method="both", 
            output_subdir=f"comparison_{Path(video_path).stem}",
            **kwargs
        )
        
        # Create comparison analysis
        comparison = {
            "video_path": video_path,
            "methods_compared": methods,
            "results_by_method": {},
            "comparison_metrics": {}
        }
        
        for method in methods:
            if method in results["embeddings"]:
                comparison["results_by_method"][method] = results["embeddings"][method]
                if method in results["analysis"]:
                    comparison["results_by_method"][method]["analysis"] = results["analysis"][method]
        
        # Add comparison metrics (could be expanded)
        if len(comparison["results_by_method"]) >= 2:
            # Simple comparison metrics
            comparison["comparison_metrics"] = self._compute_comparison_metrics(
                comparison["results_by_method"]
            )
        
        return comparison
    
    def _compute_comparison_metrics(self, results_by_method: Dict) -> Dict:
        """Compute basic comparison metrics between methods."""
        metrics = {}
        
        # Get embedding dimensions
        for method, result in results_by_method.items():
            if "embedding_shape" in result:
                metrics[f"{method}_embedding_dim"] = result["embedding_shape"][1]
        
        # Add more sophisticated metrics here as needed
        
        return metrics
    
    def _save_summary(self, results: Dict, summary_path: Path):
        """Save analysis summary to JSON file."""
        import json
        
        # Create a JSON-serializable summary
        summary = {
            "video_path": results["video_path"],
            "num_elements": results["summary"]["num_elements"],
            "methods_used": results["summary"]["methods_used"],
            "analyses_completed": results["summary"]["analyses_completed"],
            "output_directory": results["output_directory"],
            "csv_path": results["csv_path"]
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary saved to: {summary_path}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status and configuration."""
        return {
            "version": "0.1.0",
            "device": self.device,
            "output_dir": str(self.output_dir),
            "models_dir": self.models_dir,
            "processors_loaded": {
                "action": self._action_processor is not None,
                "gnn": self._gnn_processor is not None,
                "naive": self._naive_processor is not None,
                "hmm": self._hmm_analyzer is not None
            }
        }