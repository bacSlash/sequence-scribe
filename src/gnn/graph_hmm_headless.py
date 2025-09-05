import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import pickle
from tqdm import tqdm
import pandas as pd
import matplotlib.patches as mpatches
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import combinations
import glob

def load_gnn_embeddings(npz_file: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    
    try:
        # Load the NPZ file
        data = np.load(npz_file, allow_pickle=True)
        
        # Extract embeddings and image names
        embeddings_dict = {}
        for key in data.files:
            embeddings_dict[key] = data[key]
        
        # Load metadata file to get proper ordering of images
        metadata_file = os.path.join(os.path.dirname(npz_file), "embeddings_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            image_names = metadata.get("image_names", list(embeddings_dict.keys()))
        else:
            # If metadata doesn't exist, use dictionary keys
            image_names = list(embeddings_dict.keys())
            
        # Ensure image_names are strings
        image_names = [str(name) for name in image_names]
        
        print(f"Loaded {len(embeddings_dict)} embeddings with dimension {next(iter(embeddings_dict.values())).shape}")
        print(f"Image names: {image_names[:5]}... (total: {len(image_names)})")
        
        return embeddings_dict, image_names
    
    except Exception as e:
        print(f"Error loading GNN embeddings from {npz_file}: {e}")
        raise

def create_sequential_embeddings(
    embeddings_dict: Dict[str, np.ndarray], 
    image_names: List[str]
) -> np.ndarray:
    
    # Gather embeddings in the correct sequence
    embeddings_list = []
    
    for img_name in image_names:
        if img_name in embeddings_dict:
            embeddings_list.append(embeddings_dict[img_name])
        else:
            print(f"Warning: No embedding found for {img_name}")
    
    # Convert to numpy array
    return np.array(embeddings_list)

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    
    # Add small epsilon to avoid division by zero
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    return embeddings / norms

def detect_outliers(embeddings: np.ndarray, z_threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    
    # Calculate Z-scores for each dimension
    z_scores = np.abs(stats.zscore(embeddings, axis=0))
    
    # Find outliers across any dimension
    outlier_mask = np.any(z_scores > z_threshold, axis=1)
    outlier_indices = np.where(outlier_mask)[0]
    
    # Create cleaned embeddings (with outliers set to mean values)
    cleaned_embeddings = embeddings.copy()
    if len(outlier_indices) > 0:
        # Replace outliers with mean values
        dimension_means = np.mean(embeddings[~outlier_mask], axis=0)
        cleaned_embeddings[outlier_mask] = dimension_means
    
    return outlier_indices, cleaned_embeddings

def plot_embedding_distributions(
    embeddings: np.ndarray, 
    output_path: Optional[str] = None,
    n_dims: int = 5,  # Plot first n dimensions
    title: str = "Embedding Distributions"
) -> None:
    
    # Select a subset of dimensions to plot
    n_dims = min(n_dims, embeddings.shape[1])
    
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3*n_dims))
    
    # Handle single dimension case
    if n_dims == 1:
        axes = [axes]
    
    for i in range(n_dims):
        sns.histplot(embeddings[:, i], kde=True, ax=axes[i])
        axes[i].set_title(f"Dimension {i+1}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
    
    plt.tight_layout()
    fig.suptitle(title, fontsize=16, y=1.02)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved embedding distributions plot to {output_path}")
    else:
        plt.show()
    plt.close()

def reduce_dimensions(
    train_embeddings: np.ndarray, 
    test_embeddings: np.ndarray, 
    n_components: int = 20,
    apply_standard_scaling: bool = True
) -> Tuple[np.ndarray, np.ndarray, Any, Any]:
    
    # Ensure n_components is not larger than the input dimension
    n_components = min(n_components, train_embeddings.shape[1])
    
    # Fit PCA on training data
    pca = PCA(n_components=n_components)
    pca.fit(train_embeddings)
    
    # Transform both training and test data
    train_reduced = pca.transform(train_embeddings)
    test_reduced = pca.transform(test_embeddings)
    
    # Apply standard scaling if requested
    scaler = None
    if apply_standard_scaling:
        scaler = StandardScaler()
        train_reduced = scaler.fit_transform(train_reduced)
        test_reduced = scaler.transform(test_reduced)
    
    # Calculate explained variance
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    print(f"Reduced dimensions from {train_embeddings.shape[1]} to {n_components}")
    print(f"Explained variance: {explained_variance:.2f}%")
    
    return train_reduced, test_reduced, pca, scaler

def train_hmm(
    embeddings: np.ndarray, 
    n_states: int = 3, 
    covariance_type: str = "diag", 
    n_iter: int = 100, 
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[hmm.GaussianHMM, float]:
    
    # Create and train the model
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        verbose=verbose
    )
    
    # Fit the model and get log likelihood
    model.fit(embeddings)
    log_likelihood = model.score(embeddings)
    
    return model, log_likelihood

def evaluate_hmm(
    model: hmm.GaussianHMM, 
    embeddings: np.ndarray, 
    frame_names: List[str]
) -> Dict[str, Any]:
    
    # Get the most likely state sequence
    hidden_states = model.predict(embeddings)
    
    # Get log likelihood
    log_likelihood = model.score(embeddings)
    
    # Get transition matrix
    transition_matrix = model.transmat_
    
    # Get state means
    state_means = model.means_
    
    # Create frame to state mapping
    frame_to_state = {frame: state for frame, state in zip(frame_names, hidden_states)}
    
    # Calculate average time spent in each state
    state_counts = np.bincount(hidden_states, minlength=model.n_components)
    state_proportions = state_counts / len(hidden_states)
    
    # Calculate state transition frequencies
    state_transitions = []
    for i in range(len(hidden_states) - 1):
        state_transitions.append((hidden_states[i], hidden_states[i+1]))
        
    transition_counts = {}
    for from_state, to_state in state_transitions:
        key = (from_state, to_state)
        transition_counts[key] = transition_counts.get(key, 0) + 1
    
    # Calculate state durations (how long each state lasts)
    state_durations = []
    current_state = hidden_states[0]
    current_duration = 1
    
    for i in range(1, len(hidden_states)):
        if hidden_states[i] == current_state:
            current_duration += 1
        else:
            state_durations.append((current_state, current_duration))
            current_state = hidden_states[i]
            current_duration = 1
            
    # Add the last sequence
    state_durations.append((current_state, current_duration))
    
    # Organize durations by state
    durations_by_state = {}
    for state, duration in state_durations:
        if state not in durations_by_state:
            durations_by_state[state] = []
        durations_by_state[state].append(duration)
    
    # Return evaluation metrics
    return {
        "log_likelihood": log_likelihood,
        "hidden_states": hidden_states,
        "transition_matrix": transition_matrix,
        "state_means": state_means,
        "frame_to_state": frame_to_state,
        "state_proportions": state_proportions,
        "transition_counts": transition_counts,
        "state_durations": durations_by_state
    }

def plot_transition_matrix(
    transition_matrix: np.ndarray, 
    output_path: Optional[str] = None,
    title: str = "HMM Transition Matrix"
) -> None:
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        transition_matrix, 
        annot=True, 
        fmt='.2f', 
        cmap='viridis', 
        xticklabels=range(transition_matrix.shape[1]),
        yticklabels=range(transition_matrix.shape[0])
    )
    plt.xlabel("Next State")
    plt.ylabel("Current State")
    plt.title(title)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved transition matrix plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_state_sequence(
    hidden_states: np.ndarray, 
    frame_names: List[str], 
    output_path: Optional[str] = None,
    title: str = "HMM State Sequence"
) -> None:
    
    plt.figure(figsize=(12, 6))
    
    # Extract frame indices for x-axis if frame names have format "frame_XXXX.jpg"
    try:
        x = [int(name.split('_')[1].split('.')[0]) for name in frame_names]
    except (IndexError, ValueError):
        x = range(len(hidden_states))
    
    plt.plot(x, hidden_states, 'o-', markersize=8)
    
    # Show a reasonable number of x-ticks
    if len(x) > 10:
        step = max(1, len(x) // 10)
        plt.xticks(x[::step], rotation=45, ha='right')
    else:
        plt.xticks(x, rotation=45, ha='right')
        
    plt.yticks(range(max(hidden_states) + 1))
    plt.grid(alpha=0.3)
    plt.xlabel("Frame")
    plt.ylabel("State")
    plt.title(title)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved state sequence plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_state_means(
    state_means: np.ndarray, 
    output_path: Optional[str] = None,
    title: str = "HMM State Means"
) -> None:
    
    # If there are too many dimensions, use PCA to reduce to 2D
    if state_means.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_means = pca.fit_transform(state_means)
    else:
        reduced_means = state_means
    
    plt.figure(figsize=(10, 8))
    plt.scatter(
        reduced_means[:, 0], 
        reduced_means[:, 1], 
        c=range(len(state_means)), 
        cmap='viridis', 
        s=100, 
        alpha=0.8
    )
    
    # Add state labels
    for i, (x, y) in enumerate(reduced_means):
        plt.annotate(f"State {i}", (x, y), fontsize=12, ha='center')
    
    plt.grid(alpha=0.3)
    plt.title(title)
    
    if state_means.shape[1] > 2:
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
    else:
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved state means plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_state_durations(
    state_durations: Dict[int, List[int]],
    output_path: Optional[str] = None,
    title: str = "State Durations"
) -> None:
    
    num_states = len(state_durations)
    fig, axes = plt.subplots(1, num_states, figsize=(num_states * 4, 5))
    
    # Handle single state case
    if num_states == 1:
        axes = [axes]
    
    for i, state in enumerate(sorted(state_durations.keys())):
        durations = state_durations[state]
        axes[i].hist(durations, bins=min(20, max(5, len(durations) // 3)), alpha=0.7)
        axes[i].set_title(f"State {state}")
        axes[i].set_xlabel("Duration (frames)")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    fig.suptitle(title, fontsize=16, y=1.05)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved state durations plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_pca_variance(
    pca: PCA,
    output_path: Optional[str] = None,
    title: str = "PCA Explained Variance"
) -> None:
    
    plt.figure(figsize=(10, 6))
    
    # Plot explained variance ratio
    plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-', markersize=8)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% explained variance')
    
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved PCA variance plot to {output_path}")
    else:
        plt.show()
    plt.close()

def plot_hmm_graph(
    transition_matrix: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "HMM Markov Chain",
    min_prob: float = 0.01,  # Minimum probability to show an edge
    node_size: int = 2000,
    arrow_size: int = 20,
    layout: str = "spring"  # Options: spring, circular, spectral
) -> None:
    """
    Plot a directed graph visualization of the HMM transition matrix.
    
    Args:
        transition_matrix: The transition matrix of the HMM
        output_path: Path to save the plot
        title: Title of the plot
        min_prob: Minimum probability to display an edge
        node_size: Size of nodes in the graph
        arrow_size: Size of arrows in the graph
        layout: Graph layout algorithm
    """
    import networkx as nx
    import matplotlib.patches as patches
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes (states)
    n_states = transition_matrix.shape[0]
    for i in range(n_states):
        G.add_node(i, label=f"State {i}")
    
    # Add edges with weights (transition probabilities)
    for i in range(n_states):
        for j in range(n_states):
            prob = transition_matrix[i, j]
            if prob > min_prob:  # Only show significant transitions
                G.add_edge(i, j, weight=prob, label=f"{prob:.2f}")
    
    # Create figure with specific axes layout
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Position nodes
    if layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:  # Default to spring layout
        pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)  # Added seed for reproducibility
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, 
                          node_color="#8ecae6", alpha=1.0, ax=ax)  # Changed to more vibrant node color
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight="bold", ax=ax)  # Increased font size
    
    # Define edge colors and widths based on weight
    edge_colors = [G[u][v]['weight'] for u, v in G.edges() if u != v]  # Exclude self-loops
    # Make edges thicker for better visibility
    edge_widths = [G[u][v]['weight'] * 10 for u, v in G.edges() if u != v]  # Exclude self-loops
    
    # Create edge list excluding self-loops
    edges_no_selfloops = [(u, v) for u, v in G.edges() if u != v]
    
    # Use a more vibrant colormap
    cmap = plt.cm.YlOrRd  # Changed to YlOrRd which has more contrast
    vmin = min(edge_colors) if edge_colors else 0
    vmax = max(edge_colors) if edge_colors else 1
    
    # Draw edges with increased contrast and visibility - only non-self-loops
    if edges_no_selfloops:
        edges = nx.draw_networkx_edges(G, pos, edgelist=edges_no_selfloops, width=edge_widths, 
                                      edge_color=edge_colors, edge_cmap=cmap,
                                      edge_vmin=vmin, edge_vmax=vmax,
                                      arrowsize=arrow_size+5,  # Increased arrow size
                                      connectionstyle='arc3,rad=0.2',  # More curved edges
                                      arrowstyle='-|>',
                                      ax=ax)
    
    # Add edge labels (probabilities) with a background for better readability
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges() if u != v}  # Skip self-loops here
    # Add white background to edge labels
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=11, 
                                bbox=bbox_props, font_weight='bold', ax=ax)
    
    # Add colorbar with improved visibility
    if edge_colors:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Transition Probability", fontsize=14, fontweight='bold')  # Increased font size
        cbar.ax.tick_params(labelsize=12)  # Larger tick labels
    
    # Self-loops (transitions to same state) with improved styling and a single text label
    for i in range(n_states):
        if G.has_edge(i, i):
            self_loop_weight = G[i][i]['weight']
            rad = 0.3
            
            # Create a looped arrow with more prominent color
            color_val = cmap(self_loop_weight)
            
            arrow = patches.FancyArrowPatch(
                pos[i], pos[i],
                connectionstyle=f'arc3,rad={rad}',
                arrowstyle='-|>',
                mutation_scale=25,  # Increased size
                lw=self_loop_weight*12,  # Thicker line
                color=color_val,
                zorder=0
            )
            ax.add_patch(arrow)
            
            # Add self-loop label with white background for better visibility
            # Position the label at the top of the loop
            label_pos = (pos[i][0], pos[i][1] + rad + 0.05)
            
            # Create white background for self-loop label
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            ax.text(label_pos[0], label_pos[1], f"{self_loop_weight:.2f}", 
                   fontsize=11, ha='center', va='center', 
                   bbox=bbox_props, fontweight='bold')
    
    ax.set_title(title, fontsize=18, fontweight='bold')  # Increased font size
    ax.axis('off')
    
    # Add margin to avoid cutting off node contents
    plt.tight_layout(pad=1.2)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved HMM graph visualization to {output_path}")
    else:
        plt.show()
    plt.close()

def save_evaluation_results(
    evaluation: Dict[str, Any], 
    output_path: str, 
    embedding_info: Dict[str, Any]
) -> None:
    
    # Convert numpy arrays to lists for JSON serialization
    results = {
        "embedding_info": embedding_info,
        "log_likelihood": float(evaluation["log_likelihood"]),
        "state_proportions": evaluation["state_proportions"].tolist(),
        "transition_matrix": evaluation["transition_matrix"].tolist(),
        "num_frames": len(evaluation["hidden_states"]),
        "hidden_states": evaluation["hidden_states"].tolist()
    }
    
    # Add frame to state mapping
    frame_states = {}
    for frame, state in evaluation["frame_to_state"].items():
        frame_states[str(frame)] = int(state)
    results["frame_to_state"] = frame_states
    
    # Format transition counts for readability
    transition_counts = {}
    for (from_state, to_state), count in evaluation["transition_counts"].items():
        transition_counts[f"{from_state}->{to_state}"] = count
    results["transition_counts"] = transition_counts
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved evaluation results to {output_path}")

def find_embedding_files(input_dir: str) -> List[str]:
    """Find all .npz files in the input directory."""
    pattern = os.path.join(input_dir, "*.npz")
    files = glob.glob(pattern)
    if len(files) == 0:
        raise ValueError(f"No .npz files found in {input_dir}")
    print(f"Found {len(files)} embedding files: {[os.path.basename(f) for f in files]}")
    return files

def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)

def prepare_embeddings_for_training(
    embedding_files: List[str],
    normalize: bool = True,
    detect_and_handle_outliers: bool = True,
    z_threshold: float = 3.0
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """Load and preprocess embeddings from multiple files."""
    
    processed_embeddings = {}
    
    for file_path in embedding_files:
        print(f"\nLoading and preprocessing {os.path.basename(file_path)}...")
        
        # Load embeddings
        embeddings_dict, image_names = load_gnn_embeddings(file_path)
        embeddings = create_sequential_embeddings(embeddings_dict, image_names)
        
        # Normalization step (if enabled)
        if normalize:
            embeddings = normalize_embeddings(embeddings)
        
        # Outlier detection (if enabled)
        if detect_and_handle_outliers:
            outlier_indices, cleaned_embeddings = detect_outliers(embeddings, z_threshold)
            if len(outlier_indices) > 0:
                print(f"Detected and cleaned {len(outlier_indices)} outliers")
                embeddings = cleaned_embeddings
        
        processed_embeddings[file_path] = (embeddings, image_names)
        print(f"Processed embeddings shape: {embeddings.shape}")
    
    return processed_embeddings

def cross_validate_hmm(
    embedding_files: List[str],
    output_dir: str,
    n_states: int = 3,
    covariance_type: str = "diag",
    n_iter: int = 100,
    pca_components: int = 20,
    normalize: bool = True,
    apply_standard_scaling: bool = True,
    detect_and_handle_outliers: bool = True,
    z_threshold: float = 3.0,
    test_weight: float = 0.7,
    train_weight: float = 0.3
) -> Dict[str, Any]:
    """Perform 3-fold cross-validation on embedding files without merging datasets."""
    
    if len(embedding_files) != 3:
        raise ValueError(f"Expected exactly 3 embedding files, got {len(embedding_files)}")
    
    print(f"\nStarting 3-fold cross-validation...")
    print(f"Performance metric: {test_weight:.1f} * test_ll + {train_weight:.1f} * train_ll")
    print("Training separate models on each training file, then averaging results")
    
    # Prepare embeddings
    processed_embeddings = prepare_embeddings_for_training(
        embedding_files, normalize, detect_and_handle_outliers, z_threshold
    )
    
    # Generate all possible combinations of 2 files for training, 1 for testing
    file_combinations = list(combinations(embedding_files, 2))
    
    cv_results = []
    
    for fold_idx, train_files in enumerate(file_combinations):
        test_file = [f for f in embedding_files if f not in train_files][0]
        
        print(f"\n=== FOLD {fold_idx + 1}/3 ===")
        print(f"Training on: {[os.path.basename(f) for f in train_files]} (separately)")
        print(f"Testing on: {os.path.basename(test_file)}")
        
        # Create fold output directory
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Get test data
        test_embeddings, test_frame_names = processed_embeddings[test_file]
        
        # Train separate models on each training file
        train_results = []
        trained_models = []
        
        for train_idx, train_file in enumerate(train_files):
            train_embeddings, train_frame_names = processed_embeddings[train_file]
            
            print(f"\n--- Training Model {train_idx + 1} on {os.path.basename(train_file)} ---")
            print(f"Training data shape: {train_embeddings.shape}")
            
            # Apply dimensionality reduction (fit PCA on this training set)
            # We need a dummy test set for the reduce_dimensions function, use a small subset of train data
            dummy_test = train_embeddings[:min(10, len(train_embeddings))]
            train_reduced, _, pca_model, scaler = reduce_dimensions(
                train_embeddings=train_embeddings,
                test_embeddings=dummy_test,
                n_components=pca_components,
                apply_standard_scaling=apply_standard_scaling
            )
            
            # Apply the same PCA transformation to test data
            test_reduced = pca_model.transform(test_embeddings)
            if scaler is not None:
                test_reduced = scaler.transform(test_reduced)
            
            # Train HMM
            model, train_log_likelihood = train_hmm(
                embeddings=train_reduced,
                n_states=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
                verbose=False
            )
            
            # Evaluate on test data
            test_log_likelihood = model.score(test_reduced)
            
            print(f"Model {train_idx + 1} - Train LL: {train_log_likelihood:.2f}, Test LL: {test_log_likelihood:.2f}")
            
            train_results.append({
                "train_file": train_file,
                "train_ll": train_log_likelihood,
                "test_ll": test_log_likelihood,
                "model": model,
                "pca_model": pca_model,
                "scaler": scaler,
                "train_embeddings": train_embeddings,
                "train_frame_names": train_frame_names,
                "train_reduced": train_reduced,
                "test_reduced": test_reduced
            })
            trained_models.append(model)
        
        # Average the results from both training models
        avg_train_ll = np.mean([r["train_ll"] for r in train_results])
        avg_test_ll = np.mean([r["test_ll"] for r in train_results])
        combined_score = test_weight * avg_test_ll + train_weight * avg_train_ll
        
        print(f"\n--- Fold {fold_idx + 1} Summary ---")
        print(f"Average Train Log Likelihood: {avg_train_ll:.2f}")
        print(f"Average Test Log Likelihood: {avg_test_ll:.2f}")
        print(f"Combined Score: {combined_score:.2f}")
        
        # Store fold results
        fold_result = {
            "fold": fold_idx + 1,
            "train_files": [os.path.basename(f) for f in train_files],
            "test_file": os.path.basename(test_file),
            "train_log_likelihood": float(avg_train_ll),
            "test_log_likelihood": float(avg_test_ll),
            "combined_score": float(combined_score),
            "individual_train_results": [
                {
                    "file": os.path.basename(r["train_file"]),
                    "train_ll": float(r["train_ll"]),
                    "test_ll": float(r["test_ll"])
                } for r in train_results
            ],
            "train_frames": sum(len(r["train_frame_names"]) for r in train_results),
            "test_frames": len(test_frame_names)
        }
        cv_results.append(fold_result)
        
        # Select the better performing model for visualization (higher test LL)
        best_model_idx = np.argmax([r["test_ll"] for r in train_results])
        best_result = train_results[best_model_idx]
        
        print(f"Using Model {best_model_idx + 1} for fold visualizations (better test performance)")
        
        # Generate fold plots using the best model
        model = best_result["model"]
        train_reduced = best_result["train_reduced"] 
        test_reduced = best_result["test_reduced"]
        train_frame_names = best_result["train_frame_names"]
        
        # Transition matrix
        transition_plot = os.path.join(fold_dir, "transition_matrix.png")
        plot_transition_matrix(
            model.transmat_, 
            transition_plot,
            title=f"Fold {fold_idx + 1} - HMM Transition Matrix (Best Model)"
        )
        
        # Markov chain graph
        markov_graph = os.path.join(fold_dir, "markov_chain_graph.png")
        plot_hmm_graph(
            model.transmat_,
            markov_graph,
            title=f"Fold {fold_idx + 1} - HMM Markov Chain (Best Model)"
        )
        
        # State sequence for training data (best model)
        train_evaluation = evaluate_hmm(model, train_reduced, train_frame_names)
        train_sequence_plot = os.path.join(fold_dir, "train_state_sequence.png")
        plot_state_sequence(
            train_evaluation["hidden_states"], 
            train_frame_names, 
            train_sequence_plot,
            title=f"Fold {fold_idx + 1} - Train State Sequence (Best Model)"
        )
        
        # State sequence for test data
        test_evaluation = evaluate_hmm(model, test_reduced, test_frame_names)
        test_sequence_plot = os.path.join(fold_dir, "test_state_sequence.png")
        plot_state_sequence(
            test_evaluation["hidden_states"], 
            test_frame_names, 
            test_sequence_plot,
            title=f"Fold {fold_idx + 1} - Test State Sequence (Best Model)"
        )
        
        # State means
        means_plot = os.path.join(fold_dir, "state_means.png")
        plot_state_means(
            model.means_, 
            means_plot,
            title=f"Fold {fold_idx + 1} - HMM State Means (Best Model)"
        )
        
        # Save detailed fold information
        fold_info = {
            "fold": fold_idx + 1,
            "train_files": [os.path.basename(f) for f in train_files],
            "test_file": os.path.basename(test_file),
            "n_states": n_states,
            "covariance_type": covariance_type,
            "pca_components": pca_components,
            "normalization_applied": normalize,
            "standard_scaling_applied": apply_standard_scaling,
            "outlier_detection_applied": detect_and_handle_outliers,
            "individual_models": fold_result["individual_train_results"],
            "averaged_results": {
                "train_ll": float(avg_train_ll),
                "test_ll": float(avg_test_ll),
                "combined_score": float(combined_score)
            },
            "best_model_for_visualization": {
                "model_index": best_model_idx + 1,
                "train_file": os.path.basename(best_result["train_file"]),
                "test_ll": float(best_result["test_ll"])
            }
        }
        
        # Save fold evaluation results (using best model)
        train_results_file = os.path.join(fold_dir, "train_evaluation_results.json")
        save_evaluation_results(train_evaluation, train_results_file, {**fold_info, "dataset": "train"})
        
        test_results_file = os.path.join(fold_dir, "test_evaluation_results.json")
        save_evaluation_results(test_evaluation, test_results_file, {**fold_info, "dataset": "test"})
        
        # Save fold summary
        fold_summary_file = os.path.join(fold_dir, "fold_summary.json")
        with open(fold_summary_file, 'w') as f:
            json.dump(fold_result, f, indent=2)
    
    # Find best performing fold
    best_fold = max(cv_results, key=lambda x: x["combined_score"])
    best_fold_idx = best_fold["fold"] - 1
    
    print(f"\n=== CROSS-VALIDATION RESULTS ===")
    for result in cv_results:
        print(f"Fold {result['fold']}: Combined Score = {result['combined_score']:.2f} "
              f"(Avg Train: {result['train_log_likelihood']:.2f}, Avg Test: {result['test_log_likelihood']:.2f})")
    
    print(f"\nBest performing fold: {best_fold['fold']} with combined score: {best_fold['combined_score']:.2f}")
    
    # Save cross-validation summary
    cv_summary = {
        "cross_validation_results": cv_results,
        "best_fold": best_fold,
        "performance_weights": {"test_weight": test_weight, "train_weight": train_weight},
        "methodology": "Separate models trained on each training file, results averaged",
        "hyperparameters": {
            "n_states": n_states,
            "covariance_type": covariance_type,
            "pca_components": pca_components,
            "normalization_applied": normalize,
            "standard_scaling_applied": apply_standard_scaling,
            "outlier_detection_applied": detect_and_handle_outliers
        }
    }
    
    cv_summary_file = os.path.join(output_dir, "cross_validation_summary.json")
    with open(cv_summary_file, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    return {
        "cv_results": cv_results,
        "best_fold": best_fold,
        "best_fold_idx": best_fold_idx,
        "processed_embeddings": processed_embeddings
    }

def train_final_model(
    embedding_files: List[str],
    cv_results: Dict[str, Any],
    output_dir: str,
    n_states: int = 3,
    covariance_type: str = "diag",
    n_iter: int = 100,
    pca_components: int = 20,
    normalize: bool = True,
    apply_standard_scaling: bool = True,
    detect_and_handle_outliers: bool = True,
    z_threshold: float = 3.0
) -> Dict[str, Any]:
    """Train final model on the largest embedding file using best configuration."""
    
    print(f"\n=== TRAINING FINAL MODEL ===")
    
    # Find the largest embedding file
    file_sizes = [(f, get_file_size(f)) for f in embedding_files]
    largest_file = max(file_sizes, key=lambda x: x[1])[0]
    
    print(f"Training final model on largest file: {os.path.basename(largest_file)}")
    print(f"File size: {get_file_size(largest_file) / (1024*1024):.2f} MB")
    
    # Create final model output directory
    final_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    
    # Load and preprocess the largest file
    embeddings_dict, image_names = load_gnn_embeddings(largest_file)
    embeddings = create_sequential_embeddings(embeddings_dict, image_names)
    
    # Plot original embedding distributions
    orig_dist_plot = os.path.join(final_dir, "original_embedding_distributions.png")
    plot_embedding_distributions(embeddings, orig_dist_plot, title="Final Model - Original GNN Embedding Distributions")
    
    # Apply same preprocessing as cross-validation
    if normalize:
        print("Normalizing embeddings...")
        embeddings = normalize_embeddings(embeddings)
        norm_dist_plot = os.path.join(final_dir, "normalized_embedding_distributions.png")
        plot_embedding_distributions(embeddings, norm_dist_plot, title="Final Model - Normalized GNN Embedding Distributions")
    
    if detect_and_handle_outliers:
        print(f"Detecting outliers...")
        outlier_indices, cleaned_embeddings = detect_outliers(embeddings, z_threshold)
        if len(outlier_indices) > 0:
            print(f"Detected and cleaned {len(outlier_indices)} outliers")
            embeddings = cleaned_embeddings
            cleaned_dist_plot = os.path.join(final_dir, "cleaned_embedding_distributions.png")
            plot_embedding_distributions(embeddings, cleaned_dist_plot, title="Final Model - Cleaned GNN Embedding Distributions")
    
    # Split data for final training/testing
    split_index = int(len(embeddings) * (2/3))
    train_embeddings = embeddings[:split_index]
    train_frames = image_names[:split_index]
    test_embeddings = embeddings[split_index:]
    test_frames = image_names[split_index:]
    
    print(f"Final split: {len(train_embeddings)} training, {len(test_embeddings)} testing frames")
    
    # Apply dimensionality reduction
    train_reduced, test_reduced, pca_model, scaler = reduce_dimensions(
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        n_components=pca_components,
        apply_standard_scaling=apply_standard_scaling
    )
    
    # Plot PCA explained variance
    pca_variance_plot = os.path.join(final_dir, "pca_explained_variance.png")
    plot_pca_variance(pca_model, pca_variance_plot, title="Final Model - PCA Explained Variance")
    
    # Plot reduced embedding distributions
    reduced_dist_plot = os.path.join(final_dir, "reduced_embedding_distributions.png")
    plot_embedding_distributions(
        train_reduced, reduced_dist_plot, 
        n_dims=min(5, train_reduced.shape[1]),
        title="Final Model - Reduced GNN Embedding Distributions" + (" (with Standard Scaling)" if apply_standard_scaling else "")
    )
    
    # Train final HMM
    print(f"Training final HMM with {n_states} states...")
    final_model, train_log_likelihood = train_hmm(
        embeddings=train_reduced,
        n_states=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        verbose=True
    )
    
    # Evaluate final model
    train_evaluation = evaluate_hmm(final_model, train_reduced, train_frames)
    test_evaluation = evaluate_hmm(final_model, test_reduced, test_frames)
    test_log_likelihood = test_evaluation["log_likelihood"]
    
    print(f"Final model - Train Log Likelihood: {train_log_likelihood:.2f}")
    print(f"Final model - Test Log Likelihood: {test_log_likelihood:.2f}")
    
    # Generate final model plots
    # Transition matrix
    transition_plot = os.path.join(final_dir, "transition_matrix.png")
    plot_transition_matrix(
        final_model.transmat_, 
        transition_plot,
        title=f"Final Model - HMM Transition Matrix (n_states={n_states})"
    )
    
    # Markov chain graph
    markov_graph = os.path.join(final_dir, "markov_chain_graph.png")
    plot_hmm_graph(
        final_model.transmat_,
        markov_graph,
        title=f"Final Model - HMM Markov Chain (n_states={n_states})"
    )
    
    # State means
    means_plot = os.path.join(final_dir, "state_means.png")
    plot_state_means(
        final_model.means_, 
        means_plot,
        title=f"Final Model - HMM State Means (n_states={n_states})"
    )
    
    # State sequences
    train_sequence_plot = os.path.join(final_dir, "train_state_sequence.png")
    plot_state_sequence(
        train_evaluation["hidden_states"], 
        train_frames, 
        train_sequence_plot,
        title=f"Final Model - Training State Sequence (n_states={n_states})"
    )
    
    test_sequence_plot = os.path.join(final_dir, "test_state_sequence.png")
    plot_state_sequence(
        test_evaluation["hidden_states"], 
        test_frames, 
        test_sequence_plot,
        title=f"Final Model - Test State Sequence (n_states={n_states})"
    )
    
    # State durations
    durations_plot = os.path.join(final_dir, "state_durations.png")
    plot_state_durations(
        train_evaluation["state_durations"],
        durations_plot,
        title=f"Final Model - State Durations (n_states={n_states})"
    )
    
    # Final model info
    final_model_info = {
        "training_file": os.path.basename(largest_file),
        "file_size_mb": get_file_size(largest_file) / (1024*1024),
        "n_states": n_states,
        "covariance_type": covariance_type,
        "original_embedding_shape": embeddings.shape,
        "reduced_embedding_shape": train_reduced.shape,
        "pca_components": pca_components,
        "pca_explained_variance": float(sum(pca_model.explained_variance_ratio_) * 100),
        "normalization_applied": normalize,
        "standard_scaling_applied": apply_standard_scaling,
        "outlier_detection_applied": detect_and_handle_outliers,
        "outliers_detected": len(outlier_indices) if detect_and_handle_outliers else 0,
        "train_frames": len(train_frames),
        "test_frames": len(test_frames),
        "train_log_likelihood": float(train_log_likelihood),
        "test_log_likelihood": float(test_log_likelihood),
        "cross_validation_best_fold": cv_results["best_fold"]["fold"],
        "cross_validation_best_score": cv_results["best_fold"]["combined_score"]
    }
    
    # Save final model evaluation results
    train_results_file = os.path.join(final_dir, "train_evaluation_results.json")
    save_evaluation_results(train_evaluation, train_results_file, {**final_model_info, "dataset": "train"})
    
    test_results_file = os.path.join(final_dir, "test_evaluation_results.json")
    save_evaluation_results(test_evaluation, test_results_file, {**final_model_info, "dataset": "test"})
    
    # Save final model summary
    final_summary_file = os.path.join(final_dir, "final_model_summary.json")
    with open(final_summary_file, 'w') as f:
        json.dump(final_model_info, f, indent=2)
    
    return final_model_info

def generate_cross_validation_plots(output_dir: str, cv_results: List[Dict]) -> None:
    """Generate summary plots for cross-validation results."""
    
    print("\nGenerating cross-validation summary plots...")
    
    # Performance comparison plot
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Combined scores
    plt.subplot(1, 3, 1)
    folds = [r["fold"] for r in cv_results]
    combined_scores = [r["combined_score"] for r in cv_results]
    colors = ['red' if score == max(combined_scores) else 'blue' for score in combined_scores]
    
    bars = plt.bar(folds, combined_scores, color=colors, alpha=0.7)
    plt.xlabel("Fold")
    plt.ylabel("Combined Score")
    plt.title("Cross-Validation Performance")
    plt.grid(alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, combined_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 2: Train vs Test performance
    plt.subplot(1, 3, 2)
    train_lls = [r["train_log_likelihood"] for r in cv_results]
    test_lls = [r["test_log_likelihood"] for r in cv_results]
    
    x = np.arange(len(folds))
    width = 0.35
    
    plt.bar(x - width/2, train_lls, width, label='Train LL', alpha=0.7)
    plt.bar(x + width/2, test_lls, width, label='Test LL', alpha=0.7)
    
    plt.xlabel("Fold")
    plt.ylabel("Log Likelihood")
    plt.title("Train vs Test Performance")
    plt.xticks(x, folds)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Subplot 3: Data distribution
    plt.subplot(1, 3, 3)
    train_frames = [r["train_frames"] for r in cv_results]
    test_frames = [r["test_frames"] for r in cv_results]
    
    plt.bar(x - width/2, train_frames, width, label='Train Frames', alpha=0.7)
    plt.bar(x + width/2, test_frames, width, label='Test Frames', alpha=0.7)
    
    plt.xlabel("Fold")
    plt.ylabel("Number of Frames")
    plt.title("Data Distribution")
    plt.xticks(x, folds)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    cv_plot_file = os.path.join(output_dir, "cross_validation_summary.png")
    plt.savefig(cv_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved cross-validation summary plot to {cv_plot_file}")

def main():
    
    parser = argparse.ArgumentParser(description="Cross-validation HMM training on multiple GNN embedding files")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing embedding NPZ files")
    parser.add_argument("--output_dir", type=str, default="cv_gnn_hmm_results",
                        help="Directory to save cross-validation results")
    parser.add_argument("--n_states", type=int, default=3,
                        help="Number of hidden states for the HMM")
    parser.add_argument("--covariance_type", type=str, default="diag",
                        choices=["spherical", "tied", "diag", "full"],
                        help="Type of covariance matrix for the HMM")
    parser.add_argument("--n_iter", type=int, default=100,
                        help="Maximum number of iterations for EM algorithm")
    parser.add_argument("--pca_components", type=int, default=20,
                        help="Number of PCA components to use")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Disable normalization of embeddings")
    parser.add_argument("--no_standard_scaling", action="store_true",
                        help="Disable standard scaling after PCA")
    parser.add_argument("--no_outlier_detection", action="store_true",
                        help="Disable outlier detection and handling")
    parser.add_argument("--z_threshold", type=float, default=3.0,
                        help="Z-score threshold for outlier detection")
    parser.add_argument("--test_weight", type=float, default=0.7,
                        help="Weight for test performance in combined metric")
    parser.add_argument("--train_weight", type=float, default=0.3,
                        help="Weight for train performance in combined metric")
    
    args = parser.parse_args()
    
    # Validate weights
    if abs(args.test_weight + args.train_weight - 1.0) > 1e-6:
        print("Warning: test_weight + train_weight should equal 1.0")
        print(f"Current: {args.test_weight} + {args.train_weight} = {args.test_weight + args.train_weight}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find embedding files
    try:
        embedding_files = find_embedding_files(args.input_dir)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    if len(embedding_files) != 3:
        print(f"Error: Expected exactly 3 embedding files, found {len(embedding_files)}")
        print("Files found:", [os.path.basename(f) for f in embedding_files])
        return
    
    print(f"\nStarting cross-validation HMM analysis...")
    print(f"Embedding files: {[os.path.basename(f) for f in embedding_files]}")
    print(f"Output directory: {args.output_dir}")
    
    # Perform cross-validation
    cv_results = cross_validate_hmm(
        embedding_files=embedding_files,
        output_dir=args.output_dir,
        n_states=args.n_states,
        covariance_type=args.covariance_type,
        n_iter=args.n_iter,
        pca_components=args.pca_components,
        normalize=not args.no_normalize,
        apply_standard_scaling=not args.no_standard_scaling,
        detect_and_handle_outliers=not args.no_outlier_detection,
        z_threshold=args.z_threshold,
        test_weight=args.test_weight,
        train_weight=args.train_weight
    )
    
    # Generate cross-validation summary plots
    generate_cross_validation_plots(args.output_dir, cv_results["cv_results"])
    
    # Train final model on largest file
    final_model_info = train_final_model(
        embedding_files=embedding_files,
        cv_results=cv_results,
        output_dir=args.output_dir,
        n_states=args.n_states,
        covariance_type=args.covariance_type,
        n_iter=args.n_iter,
        pca_components=args.pca_components,
        normalize=not args.no_normalize,
        apply_standard_scaling=not args.no_standard_scaling,
        detect_and_handle_outliers=not args.no_outlier_detection,
        z_threshold=args.z_threshold
    )
    
    # Print final summary
    print(f"\n{'='*50}")
    print(f"CROSS-VALIDATION HMM ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print(f"Best cross-validation fold: {cv_results['best_fold']['fold']}")
    print(f"Best combined score: {cv_results['best_fold']['combined_score']:.2f}")
    print(f"Final model trained on: {final_model_info['training_file']}")
    print(f"Final model train LL: {final_model_info['train_log_likelihood']:.2f}")
    print(f"Final model test LL: {final_model_info['test_log_likelihood']:.2f}")
    print(f"All results saved to: {args.output_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()

# Usage examples:
# python cv_gnn_hmm.py --input_dir /path/to/embeddings --output_dir cv_results
# python cv_gnn_hmm.py --input_dir ./embeddings --output_dir ./cv_results --n_states 5 --test_weight 0.8 --train_weight 0.2
