"""
Utilities module with helper functions.
"""

import os
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.decomposition import PCA

def setup_logging(level: str = "INFO", output_dir: str = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        output_dir: Directory to save log files
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("sequence_scribe")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(output_dir / "sequence_scribe.log")
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def validate_file_path(file_path: str) -> str:
    """
    Validate that a file path exists and is accessible.
    
    Args:
        file_path: Path to validate
        
    Returns:
        Validated file path
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If path is invalid
    """
    if not file_path:
        raise ValueError("File path cannot be empty")
    
    file_path = Path(file_path).resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    return str(file_path)

def create_output_dir(output_dir: str) -> str:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Directory path to create
        
    Returns:
        Created directory path
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)

def load_embeddings(embeddings_file: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load embeddings from various file formats.
    
    Args:
        embeddings_file: Path to embeddings file
        
    Returns:
        Tuple of (embeddings_array, frame_names)
    """
    embeddings_file = validate_file_path(embeddings_file)
    
    if embeddings_file.endswith('.npz'):
        data = np.load(embeddings_file, allow_pickle=True)
        
        # Handle different formats
        if 'embeddings' in data:
            embeddings = data['embeddings']
            frame_names = data.get('frame_names', [f"frame_{i}" for i in range(len(embeddings))])
        elif 'observations' in data:
            embeddings = data['observations']
            frame_names = data.get('frame_names', [f"frame_{i}" for i in range(len(embeddings))])
        else:
            # Handle GNN embedding format (dictionary of embeddings keyed by frame names)
            frame_names = list(data.files)
            embeddings = np.array([data[name] for name in frame_names])
        
        # Convert frame_names to list of strings if needed
        if isinstance(frame_names, np.ndarray):
            frame_names = [str(name) for name in frame_names]
        elif not isinstance(frame_names, list):
            frame_names = list(frame_names)
        
        return embeddings, frame_names
    
    else:
        raise ValueError(f"Unsupported file format: {embeddings_file}")

def save_embeddings(
    embeddings: np.ndarray,
    frame_names: List[str],
    output_file: str,
    metadata: Dict[str, Any] = None
):
    """
    Save embeddings to file with metadata.
    
    Args:
        embeddings: Embeddings array
        frame_names: List of frame names
        output_file: Output file path
        metadata: Optional metadata dictionary
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        frame_names=frame_names
    )
    
    # Save metadata if provided
    if metadata:
        metadata_file = output_file.with_suffix('.json')
        metadata_dict = {
            "embedding_shape": embeddings.shape,
            "num_frames": len(frame_names),
            "frame_names": frame_names,
            **metadata
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

def visualize_embeddings(
    embeddings: np.ndarray,
    frame_names: List[str] = None,
    output_path: str = None,
    title: str = "Embedding Visualization",
    method: str = "pca",
    color_by: str = "sequence",
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Create visualization of embeddings.
    
    Args:
        embeddings: Embeddings array
        frame_names: List of frame names
        output_path: Path to save visualization
        title: Plot title
        method: Dimensionality reduction method ("pca", "tsne")
        color_by: How to color points ("sequence", "random")
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if frame_names is None:
        frame_names = [f"frame_{i}" for i in range(len(embeddings))]
    
    # Apply dimensionality reduction
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        reduced_embeddings = reducer.fit_transform(embeddings)
        explained_var = reducer.explained_variance_ratio_
        xlabel = f"PC1 ({explained_var[0]:.2%} variance)"
        ylabel = f"PC2 ({explained_var[1]:.2%} variance)"
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        xlabel = "t-SNE 1"
        ylabel = "t-SNE 2"
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color assignment
    if color_by == "sequence":
        colors = range(len(embeddings))
        cmap = 'viridis'
    else:
        colors = 'blue'
        cmap = None
    
    # Scatter plot
    scatter = ax.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=colors,
        cmap=cmap,
        alpha=0.7,
        s=50
    )
    
    # Add colorbar if using sequence coloring
    if color_by == "sequence":
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Frame Sequence')
    
    # Add annotations for some points
    step = max(1, len(frame_names) // 20)  # Annotate ~20 points max
    for i in range(0, len(frame_names), step):
        frame_idx = extract_frame_number(frame_names[i]) if '_' in frame_names[i] else i
        ax.annotate(
            str(frame_idx),
            (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
            fontsize=8,
            alpha=0.8
        )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    return fig

def extract_frame_number(frame_name: str) -> int:
    """
    Extract frame number from frame name.
    
    Args:
        frame_name: Frame name string
        
    Returns:
        Frame number
    """
    try:
        # Try to extract number from patterns like "frame_0001.jpg"
        parts = frame_name.split('_')
        if len(parts) > 1:
            number_part = parts[1].split('.')[0]
            return int(number_part)
    except (ValueError, IndexError):
        pass
    
    # Fallback: try to find any number in the string
    import re
    numbers = re.findall(r'\d+', frame_name)
    if numbers:
        return int(numbers[0])
    
    return 0

def save_results(
    results: Dict[str, Any],
    output_file: str,
    format: str = "json"
):
    """
    Save analysis results to file.
    
    Args:
        results: Results dictionary
        output_file: Output file path
        format: Output format ("json", "yaml")
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Make results JSON-serializable
    serializable_results = make_json_serializable(results)
    
    if format == "json":
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    elif format == "yaml":
        import yaml
        with open(output_file, 'w') as f:
            yaml.dump(serializable_results, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

def make_json_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, '__dict__'):
        # Handle custom objects by converting to dict
        return make_json_serializable(obj.__dict__)
    else:
        return obj

def compute_similarity_matrix(embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Compute similarity matrix between embeddings.
    
    Args:
        embeddings: Embeddings array
        metric: Similarity metric ("cosine", "euclidean", "manhattan")
        
    Returns:
        Similarity matrix
    """
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
    
    if metric == "cosine":
        return cosine_similarity(embeddings)
    elif metric == "euclidean":
        # Convert distances to similarities
        distances = euclidean_distances(embeddings)
        max_dist = np.max(distances)
        return 1 - (distances / max_dist)
    elif metric == "manhattan":
        distances = manhattan_distances(embeddings)
        max_dist = np.max(distances)
        return 1 - (distances / max_dist)
    else:
        raise ValueError(f"Unknown metric: {metric}")

def plot_similarity_matrix(
    similarity_matrix: np.ndarray,
    frame_names: List[str] = None,
    output_path: str = None,
    title: str = "Similarity Matrix"
) -> plt.Figure:
    """
    Plot similarity matrix as heatmap.
    
    Args:
        similarity_matrix: Similarity matrix
        frame_names: Frame names for labels
        output_path: Path to save plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Similarity')
    
    # Set labels if provided
    if frame_names:
        # Show subset of labels to avoid clutter
        step = max(1, len(frame_names) // 10)
        indices = range(0, len(frame_names), step)
        labels = [extract_frame_number(frame_names[i]) for i in indices]
        
        ax.set_xticks(indices)
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticks(indices)
        ax.set_yticklabels(labels)
    
    ax.set_title(title)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Frame')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Similarity matrix saved to: {output_path}")
    
    return fig

def create_summary_report(
    results: Dict[str, Any],
    output_path: str,
    template: str = "default"
):
    """
    Create a summary report from analysis results.
    
    Args:
        results: Analysis results
        output_path: Output file path
        template: Report template ("default", "detailed")
    """
    output_path = Path(output_path)
    
    if template == "default":
        report = f"""
# Sequence Scribe Analysis Report

## Video Information
- **Video Path**: {results.get('video_path', 'N/A')}
- **Processing Method**: {results.get('method_used', 'N/A')}
- **Output Directory**: {results.get('output_directory', 'N/A')}

## Extraction Results
- **UI Elements Extracted**: {results.get('summary', {}).get('num_elements', 'N/A')}
- **CSV File**: {Path(results.get('csv_path', '')).name}

## Embedding Methods Used
{chr(10).join(f"- {method}" for method in results.get('summary', {}).get('methods_used', []))}

## Analysis Results
{chr(10).join(f"- {method}" for method in results.get('summary', {}).get('analyses_completed', []))}

## Detailed Results
"""
        
        # Add embedding details
        for method, emb_results in results.get('embeddings', {}).items():
            if isinstance(emb_results, dict):
                report += f"""
### {method.upper()} Embeddings
- **Status**: {emb_results.get('status', 'Unknown')}
- **Embedding Shape**: {emb_results.get('embedding_shape', 'N/A')}
- **Output Directory**: {Path(emb_results.get('output_directory', '')).name}
"""
        
        # Add analysis details
        for method, analysis_results in results.get('analysis', {}).items():
            if isinstance(analysis_results, dict):
                report += f"""
### {method.upper()} Analysis
- **Status**: {analysis_results.get('status', 'Unknown')}
- **Train Log Likelihood**: {analysis_results.get('train_log_likelihood', 'N/A')}
- **Test Log Likelihood**: {analysis_results.get('test_log_likelihood', 'N/A')}
- **Number of States**: {analysis_results.get('n_states', 'N/A')}
"""
    
    else:  # detailed template
        report = create_detailed_report(results)
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Summary report saved to: {output_path}")

def create_detailed_report(results: Dict[str, Any]) -> str:
    """Create a detailed analysis report."""
    
    report = f"""
# Detailed Sequence Scribe Analysis Report
Generated: {import_datetime().now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
This report contains detailed results from the Sequence Scribe analysis pipeline.

## Video Processing
- **Source Video**: `{results.get('video_path', 'N/A')}`
- **Processing Method**: {results.get('method_used', 'N/A')}
- **Total UI Elements**: {results.get('summary', {}).get('num_elements', 'N/A')}

## Data Files Generated
- **Elements CSV**: `{results.get('csv_path', 'N/A')}`
- **Output Directory**: `{results.get('output_directory', 'N/A')}`

## Embedding Analysis
"""
    
    # Add detailed embedding information
    for method_name, emb_data in results.get('embeddings', {}).items():
        if isinstance(emb_data, dict):
            report += f"""
### {method_name.upper()} Method Results

#### Configuration
"""
            if 'parameters' in emb_data:
                for param, value in emb_data['parameters'].items():
                    report += f"- **{param}**: {value}\n"
            
            report += f"""
#### Results
- **Status**: {emb_data.get('status', 'Unknown')}
- **Embeddings File**: `{emb_data.get('embeddings_file', 'N/A')}`
- **Embedding Dimensions**: {emb_data.get('embedding_shape', 'N/A')}
- **Output Directory**: `{emb_data.get('output_directory', 'N/A')}`
"""
    
    # Add sequence analysis results
    if results.get('analysis'):
        report += "\n## Sequence Analysis Results\n"
        
        for method_name, analysis_data in results.get('analysis', {}).items():
            if isinstance(analysis_data, dict):
                report += f"""
### {method_name.upper()} HMM Analysis

#### Model Performance
- **Training Log Likelihood**: {analysis_data.get('train_log_likelihood', 'N/A')}
- **Test Log Likelihood**: {analysis_data.get('test_log_likelihood', 'N/A')}
- **Number of States**: {analysis_data.get('n_states', 'N/A')}

#### Configuration
"""
                if 'analysis_metadata' in analysis_data:
                    params = analysis_data['analysis_metadata'].get('parameters', {})
                    for param, value in params.items():
                        report += f"- **{param}**: {value}\n"
    
    report += """
## Files and Directories
All analysis outputs have been saved to the specified output directory. Key files include:
- CSV files with extracted UI elements
- NPZ files with computed embeddings
- PNG files with visualizations
- JSON files with analysis results
- Log files with processing details

## Next Steps
1. Review the generated visualizations
2. Examine the HMM analysis results
3. Consider adjusting parameters if needed
4. Use the embeddings for further analysis
"""
    
    return report

def import_datetime():
    """Import datetime module (helper for report generation)."""
    import datetime
    return datetime

def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are available.
    
    Returns:
        Dictionary with dependency status
    """
    dependencies = {
        "torch": False,
        "torch_geometric": False,
        "transformers": False,
        "sentence_transformers": False,
        "ultralytics": False,
        "opencv": False,
        "hmmlearn": False,
        "sklearn": False,
        "matplotlib": False,
        "pandas": False,
        "numpy": False,
        "networkx": False
    }
    
    try:
        import torch
        dependencies["torch"] = True
    except ImportError:
        pass
    
    try:
        import torch_geometric
        dependencies["torch_geometric"] = True
    except ImportError:
        pass
    
    try:
        import transformers
        dependencies["transformers"] = True
    except ImportError:
        pass
    
    try:
        import sentence_transformers
        dependencies["sentence_transformers"] = True
    except ImportError:
        pass
    
    try:
        import ultralytics
        dependencies["ultralytics"] = True
    except ImportError:
        pass
    
    try:
        import cv2
        dependencies["opencv"] = True
    except ImportError:
        pass
    
    try:
        import hmmlearn
        dependencies["hmmlearn"] = True
    except ImportError:
        pass
    
    try:
        import sklearn
        dependencies["sklearn"] = True
    except ImportError:
        pass
    
    try:
        import matplotlib
        dependencies["matplotlib"] = True
    except ImportError:
        pass
    
    try:
        import pandas
        dependencies["pandas"] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies["numpy"] = True
    except ImportError:
        pass
    
    try:
        import networkx
        dependencies["networkx"] = True
    except ImportError:
        pass
    
    return dependencies

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging.
    
    Returns:
        System information dictionary
    """
    import platform
    import sys
    
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.architecture(),
        "dependencies": check_dependencies()
    }
    
    # Add GPU information if available
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        info["cuda_available"] = False
    
    return info

# Export key functions
__all__ = [
    "setup_logging",
    "validate_file_path", 
    "create_output_dir",
    "load_embeddings",
    "save_embeddings",
    "visualize_embeddings",
    "save_results",
    "make_json_serializable",
    "compute_similarity_matrix",
    "plot_similarity_matrix",
    "create_summary_report",
    "check_dependencies",
    "get_system_info",
    "extract_frame_number"
]