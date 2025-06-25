"""
Sequence Scribe - A tool for analyzing UI interaction sequences
"""

__version__ = "0.1.0"

from .core import SequenceScribe
from .processors import (
    ActionProcessor,
    GNNProcessor, 
    NaiveProcessor
)
from .models import (
    UIGraphEmbedder,
)
from .analysis import (
    HMMAnalyzer,
)
from .utils import (
    load_embeddings,
    save_results,
    visualize_embeddings
)

# Main API
def extract_sequence(
    video_path: str,
    output_dir: str = None,
    method: str = "auto",
    **kwargs
) -> dict:
    """
    Extract UI action sequences from video.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory (default: auto-generated)
        method: Analysis method ('auto', 'gnn', 'naive')
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with analysis results
        
    Example:
        >>> import sequence_scribe as ss
        >>> results = ss.extract_sequence("video.mp4", method="gnn")
        >>> print(f"Found {len(results['actions'])} actions")
    """
    scribe = SequenceScribe(output_dir=output_dir)
    return scribe.process_video(video_path, method=method, **kwargs)

def analyze_embeddings(
    embeddings_file: str,
    method: str = "hmm",
    output_dir: str = None,
    **kwargs
) -> dict:
    """
    Analyze pre-computed embeddings.
    
    Args:
        embeddings_file: Path to embeddings file (.npz)
        method: Analysis method ('hmm', 'clustering')
        output_dir: Output directory
        **kwargs: Additional parameters
        
    Returns:
        Analysis results
        
    Example:
        >>> import sequence_scribe as ss
        >>> results = ss.analyze_embeddings("embeddings.npz", method="hmm")
    """
    if method == "hmm":
        from .analysis import HMMAnalyzer
        analyzer = HMMAnalyzer(output_dir=output_dir)
        return analyzer.analyze(embeddings_file, **kwargs)
    else:
        raise ValueError(f"Unknown analysis method: {method}")

def create_graph_embeddings(
    csv_path: str,
    output_dir: str = None,
    **kwargs
) -> dict:
    """
    Create graph embeddings from UI element data.
    
    Args:
        csv_path: Path to CSV with UI elements
        output_dir: Output directory
        **kwargs: Additional parameters
        
    Returns:
        Embedding results
        
    Example:
        >>> import sequence_scribe as ss
        >>> results = ss.create_graph_embeddings("ui_elements.csv")
    """
    processor = GNNProcessor(output_dir=output_dir)
    return processor.create_embeddings(csv_path, **kwargs)

def create_text_embeddings(
    csv_path: str,
    output_dir: str = None,
    model_name: str = "all-MiniLM-L6-v2",
    **kwargs
) -> dict:
    """
    Create text embeddings from UI element data.
    
    Args:
        csv_path: Path to CSV with UI elements
        output_dir: Output directory
        model_name: SentenceTransformer model name
        **kwargs: Additional parameters
        
    Returns:
        Embedding results
        
    Example:
        >>> import sequence_scribe as ss
        >>> results = ss.create_text_embeddings("ui_elements.csv")
    """
    processor = NaiveProcessor(output_dir=output_dir)
    return processor.create_embeddings(csv_path, model_name=model_name, **kwargs)

# Convenience functions
def load_config(config_path: str) -> dict:
    """Load configuration from file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def list_models() -> dict:
    """List available models and their status."""
    from .models import ModelRegistry
    return ModelRegistry.list_available()

__all__ = [
    # Main API functions
    "extract_sequence",
    "analyze_embeddings", 
    "create_graph_embeddings",
    "create_text_embeddings",
    
    # Core classes
    "SequenceScribe",
    "ActionProcessor",
    "GNNProcessor",
    "NaiveProcessor",
    "UIGraphEmbedder",
    "HMMAnalyzer",
    
    # Utility functions
    "load_embeddings",
    "save_results",
    "visualize_embeddings",
    "load_config",
    "list_models",
]
