"""
Processor classes that wrap the original functionality into clean APIs.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# Import original functions (these will be the same as your current code)
from .legacy import action_identification
from .legacy import graph_embedding_headless
from .legacy import graph_rep_to_emb_headless
from .legacy import graph_representation_headless
from .legacy import naive_json_embedding_headless

class ActionProcessor:
    """
    Processes videos to extract UI elements and actions.
    
    This wraps the functionality from your action_identification.py
    """
    
    def __init__(self, models_dir: str = None, device: str = None, logger: logging.Logger = None):
        self.models_dir = models_dir
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize models (lazy loading)
        self._yolo_model = None
        self._caption_model_processor = None
    
    def _get_yolo_model(self):
        """Get YOLO model with lazy loading."""
        if self._yolo_model is None:
            try:
                model_path = self._find_model_path()
                from .utils import get_yolo_model
                self._yolo_model = get_yolo_model(model_path)
                self.logger.info(f"Loaded YOLO model from: {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load YOLO model: {e}")
                raise
        return self._yolo_model
    
    def _get_caption_model_processor(self):
        """Get caption model with lazy loading."""
        if self._caption_model_processor is None:
            try:
                from .utils import get_caption_model_processor
                self._caption_model_processor = get_caption_model_processor(
                    'florence2', 'Microsoft/Florence-2-base'
                )
                self.logger.info("Loaded Florence-2 caption model")
            except Exception as e:
                self.logger.error(f"Failed to load caption model: {e}")
                raise
        return self._caption_model_processor
    
    def _find_model_path(self, model_filename='best.pt', search_folder='icon_detect'):
        """Find model path in models directory."""
        if self.models_dir:
            model_path = Path(self.models_dir) / search_folder / model_filename
            if model_path.exists():
                return str(model_path)
        
        # Fallback to original search logic
        return action_identification.find_model_path(model_filename, f'weights/{search_folder}')
    
    def extract_sequence_from_video(
        self,
        video_path: str,
        output_dir: str = None,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Extract UI elements from video frames.
        
        Args:
            video_path: Path to video file
            output_dir: Output directory
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (elements_list, csv_path)
        """
        if output_dir is None:
            output_dir = Path(video_path).parent / f"{Path(video_path).stem}_analysis"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Extracting UI elements from: {video_path}")
        self.logger.info(f"Output directory: {output_dir}")
        
        # Set global models for the legacy code
        action_identification.yolo_model = self._get_yolo_model()
        action_identification.caption_model_processor = self._get_caption_model_processor()
        
        # Call the original function
        try:
            elements, csv_path = action_identification.extract_sequence_from_video(
                video_path, str(output_dir)
            )
            self.logger.info(f"Successfully extracted {len(elements)} elements")
            return elements, csv_path
        except Exception as e:
            self.logger.error(f"Failed to extract sequence: {e}")
            raise
    
    def parse_frame(
        self,
        frame_path: str,
        previous_elements: List[Dict] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Parse a single frame to extract UI elements.
        
        Args:
            frame_path: Path to frame image
            previous_elements: Previous frame elements for comparison
            **kwargs: Additional parameters
            
        Returns:
            List of UI elements
        """
        # Set global models
        action_identification.yolo_model = self._get_yolo_model()
        action_identification.caption_model_processor = self._get_caption_model_processor()
        
        try:
            elements = action_identification.parse_frame(frame_path, previous_elements)
            return elements
        except Exception as e:
            self.logger.error(f"Failed to parse frame {frame_path}: {e}")
            raise


class GNNProcessor:
    """
    Creates graph-based embeddings from UI element data.
    
    This wraps the functionality from your GNN modules.
    """
    
    def __init__(self, output_dir: str = None, device: str = None, logger: logging.Logger = None):
        self.output_dir = output_dir
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
    
    def create_embeddings(
        self,
        csv_path: str,
        output_dir: str = None,
        proximity_threshold: float = 0.2,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create graph embeddings from UI element CSV data.
        
        Args:
            csv_path: Path to CSV file with UI elements
            output_dir: Output directory
            proximity_threshold: Threshold for connecting graph nodes
            hidden_dim: Hidden dimension for GNN
            output_dim: Output embedding dimension
            num_layers: Number of GNN layers
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with embedding results
        """
        if output_dir is None:
            output_dir = Path(self.output_dir) / "gnn_embeddings"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating graph embeddings from: {csv_path}")
        self.logger.info(f"Output directory: {output_dir}")
        
        try:
            # Step 1: Create graph representations
            graph_data_dir = output_dir / "graphs"
            graph_data_dir.mkdir(exist_ok=True)
            
            self.logger.info("Step 1: Creating graph representations...")
            graph_rep_to_emb_headless.save_graphs_for_embedding(
                csv_path=csv_path,
                output_dir=str(graph_data_dir),
                proximity_threshold=proximity_threshold
            )
            
            # Step 2: Generate embeddings
            embeddings_dir = output_dir / "embeddings"
            embeddings_dir.mkdir(exist_ok=True)
            
            self.logger.info("Step 2: Generating graph embeddings...")
            
            # Create embedder
            from .models import UIGraphEmbedder
            embedder = UIGraphEmbedder(
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                device=self.device
            )
            
            # Load graphs
            graphs = graph_embedding_headless.load_graphs_from_directory(str(graph_data_dir))
            
            if not graphs:
                raise ValueError("No graphs were created from the CSV data")
            
            # Generate embeddings
            embeddings = embedder.generate_embeddings(graphs)
            
            # Save embeddings
            embedder.save_embeddings(embeddings, str(embeddings_dir))
            
            # Create visualizations
            embedder.visualize_embeddings(embeddings, str(embeddings_dir))
            
            embeddings_file = embeddings_dir / "ui_embeddings.npz"
            
            results = {
                "status": "success",
                "embeddings_file": str(embeddings_file),
                "output_directory": str(output_dir),
                "graphs_directory": str(graph_data_dir),
                "num_graphs": len(graphs),
                "embedding_shape": (len(embeddings), output_dim),
                "parameters": {
                    "proximity_threshold": proximity_threshold,
                    "hidden_dim": hidden_dim,
                    "output_dim": output_dim,
                    "num_layers": num_layers
                }
            }
            
            self.logger.info(f"Successfully created {len(embeddings)} graph embeddings")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to create graph embeddings: {e}")
            raise
    
    def create_graph_representations(
        self,
        csv_path: str,
        output_dir: str = None,
        proximity_threshold: float = 0.2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create graph representations from UI data without embeddings.
        
        Args:
            csv_path: Path to CSV file
            output_dir: Output directory
            proximity_threshold: Threshold for connecting nodes
            **kwargs: Additional parameters
            
        Returns:
            Results dictionary
        """
        if output_dir is None:
            output_dir = Path(self.output_dir) / "graphs"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating graph representations from: {csv_path}")
        
        try:
            # Create graph builder
            graph_builder = graph_representation_headless.SimpleUIGraph(
                csv_path, proximity_threshold=proximity_threshold
            )
            
            # Analyze all images
            graph_builder.analyze_sample_images(n=-1, output_dir=str(output_dir))
            
            return {
                "status": "success",
                "output_directory": str(output_dir),
                "proximity_threshold": proximity_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create graph representations: {e}")
            raise


class NaiveProcessor:
    """
    Creates text-based embeddings from UI element data.
    
    This wraps the functionality from your naive embedding module.
    """
    
    def __init__(self, output_dir: str = None, logger: logging.Logger = None):
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
    
    def create_embeddings(
        self,
        csv_path: str,
        output_dir: str = None,
        model_name: str = "all-MiniLM-L6-v2",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create text embeddings from UI element CSV data.
        
        Args:
            csv_path: Path to CSV file with UI elements
            output_dir: Output directory
            model_name: SentenceTransformer model name
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with embedding results
        """
        if output_dir is None:
            output_dir = Path(self.output_dir) / "text_embeddings"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating text embeddings from: {csv_path}")
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Output directory: {output_dir}")
        
        try:
            # Call the original processing function
            results = naive_json_embedding_headless.process_csv(
                csv_path=csv_path,
                model_name=model_name,
                output_dir=str(output_dir)
            )
            
            self.logger.info(f"Successfully created text embeddings")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to create text embeddings: {e}")
            raise


class BatchProcessor:
    """
    Process multiple videos or datasets in batch.
    """
    
    def __init__(self, output_dir: str = None, logger: logging.Logger = None):
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize sub-processors
        self.action_processor = ActionProcessor(logger=self.logger)
        self.gnn_processor = GNNProcessor(output_dir=output_dir, logger=self.logger)
        self.naive_processor = NaiveProcessor(output_dir=output_dir, logger=self.logger)
    
    def process_videos(
        self,
        video_paths: List[str],
        method: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process multiple videos.
        
        Args:
            video_paths: List of video file paths
            method: Processing method
            **kwargs: Additional parameters
            
        Returns:
            Batch results
        """
        self.logger.info(f"Processing {len(video_paths)} videos with method: {method}")
        
        results = {}
        failed = []
        
        for i, video_path in enumerate(video_paths):
            try:
                self.logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
                
                video_name = Path(video_path).stem
                video_output_dir = Path(self.output_dir) / f"batch_results" / video_name
                
                # Extract elements
                elements, csv_path = self.action_processor.extract_sequence_from_video(
                    video_path, str(video_output_dir)
                )
                
                # Create embeddings based on method
                embeddings_results = {}
                
                if method in ["gnn", "auto", "both"]:
                    try:
                        gnn_results = self.gnn_processor.create_embeddings(
                            csv_path, output_dir=str(video_output_dir / "gnn")
                        )
                        embeddings_results["gnn"] = gnn_results
                    except Exception as e:
                        self.logger.error(f"GNN processing failed for {video_path}: {e}")
                
                if method in ["naive", "auto", "both"]:
                    try:
                        naive_results = self.naive_processor.create_embeddings(
                            csv_path, output_dir=str(video_output_dir / "naive")
                        )
                        embeddings_results["naive"] = naive_results
                    except Exception as e:
                        self.logger.error(f"Naive processing failed for {video_path}: {e}")
                
                results[video_name] = {
                    "video_path": video_path,
                    "elements": elements,
                    "csv_path": csv_path,
                    "embeddings": embeddings_results,
                    "output_directory": str(video_output_dir)
                }
                
                self.logger.info(f"Successfully processed: {video_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to process {video_path}: {e}")
                failed.append({"video_path": video_path, "error": str(e)})
        
        batch_results = {
            "total_videos": len(video_paths),
            "successful": len(results),
            "failed": len(failed),
            "results": results,
            "failures": failed,
            "method_used": method
        }
        
        self.logger.info(f"Batch processing complete: {len(results)} successful, {len(failed)} failed")
        return batch_results