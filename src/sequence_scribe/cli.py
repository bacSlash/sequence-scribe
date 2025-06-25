"""
Command Line Interface for Sequence Scribe.

This provides a CLI wrapper around the Python API for users who prefer command-line usage.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any

import sequence_scribe as ss
from sequence_scribe.utils import get_system_info, check_dependencies

def create_main_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="sequence-scribe",
        description="Analyze UI interaction sequences from video recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a video with automatic method selection
  sequence-scribe process video.mp4

  # Use specific method and output directory
  sequence-scribe process video.mp4 --method gnn --output ./results

  # Analyze pre-computed embeddings
  sequence-scribe analyze embeddings.npz --method hmm --states 5

  # Create embeddings from CSV data
  sequence-scribe embed-text ui_elements.csv --model all-mpnet-base-v2

  # Compare different methods
  sequence-scribe compare video.mp4 --methods gnn naive

  # Check system status
  sequence-scribe status
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process command
    process_parser = subparsers.add_parser(
        "process", help="Process video to extract and analyze UI sequences"
    )
    add_process_arguments(process_parser)
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze pre-computed embeddings"
    )
    add_analyze_arguments(analyze_parser)
    
    # Embed commands
    embed_text_parser = subparsers.add_parser(
        "embed-text", help="Create text embeddings from UI data"
    )
    add_embed_text_arguments(embed_text_parser)
    
    embed_graph_parser = subparsers.add_parser(
        "embed-graph", help="Create graph embeddings from UI data"
    )
    add_embed_graph_arguments(embed_graph_parser)
    
    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare different analysis methods"
    )
    add_compare_arguments(compare_parser)
    
    # Utility commands
    status_parser = subparsers.add_parser(
        "status", help="Check system status and dependencies"
    )
    
    info_parser = subparsers.add_parser(
        "info", help="Show detailed system information"
    )
    
    return parser

def add_process_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the process command."""
    parser.add_argument(
        "video_path",
        help="Path to video file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory (default: auto-generated)"
    )
    parser.add_argument(
        "--method", "-m",
        choices=["auto", "gnn", "naive", "both"],
        default="auto",
        help="Analysis method (default: auto)"
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip sequence analysis step"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use for processing"
    )

def add_analyze_arguments(parser: argparse.ArgumentParser):
    """Add arguments for the analyze command."""
    parser.add_argument(
        "embeddings_file",
        help="Path to embeddings file (.npz)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory"
    )
    parser.add_argument(
        "--method",
        choices=["hmm", "clustering"],
        default="hmm",
        help="Analysis method"
    )
    parser.add_argument(
        "--states", "-s",
        type=int,
        default=3,
        help="Number of states/clusters"
    )
    parser.add_argument(
        "--covariance",
        choices=["spherical", "tied", "diag", "full"],
        default="diag",
        help="HMM covariance type"
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=20,
        help="Number of PCA components"
    )

def add_embed_text_arguments(parser: argparse.ArgumentParser):
    """Add arguments for text embedding command."""
    parser.add_argument(
        "csv_path",
        help="Path to CSV file with UI elements"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory"
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name"
    )

def add_embed_graph_arguments(parser: argparse.ArgumentParser):
    """Add arguments for graph embedding command."""
    parser.add_argument(
        "csv_path",
        help="Path to CSV file with UI elements"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory"
    )
    parser.add_argument(
        "--proximity-threshold",
        type=float,
        default=0.2,
        help="Proximity threshold for connecting nodes"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for GNN"
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=64,
        help="Output embedding dimension"
    )

def add_compare_arguments(parser: argparse.ArgumentParser):
    """Add arguments for compare command."""
    parser.add_argument(
        "video_path",
        help="Path to video file"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["gnn", "naive"],
        default=["gnn", "naive"],
        help="Methods to compare"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory"
    )

def cmd_process(args) -> int:
    """Handle the process command."""
    try:
        print(f"Processing video: {args.video_path}")
        print(f"Method: {args.method}")
        
        device = None if args.device == "auto" else args.device
        
        results = ss.extract_sequence(
            video_path=args.video_path,
            output_dir=args.output,
            method=args.method,
            analyze_sequences=not args.no_analyze,
            device=device,
            log_level=args.log_level
        )
        
        print("\n✅ Processing completed successfully!")
        print(f"📁 Results saved to: {results['output_directory']}")
        print(f"📊 UI elements extracted: {results['summary']['num_elements']}")
        print(f"🔬 Methods used: {', '.join(results['summary']['methods_used'])}")
        
        if results['summary']['analyses_completed']:
            print(f"📈 Analyses completed: {', '.join(results['summary']['analyses_completed'])}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return 1

def cmd_analyze(args) -> int:
    """Handle the analyze command."""
    try:
        print(f"Analyzing embeddings: {args.embeddings_file}")
        print(f"Method: {args.method}")
        
        if args.method == "hmm":
            results = ss.analyze_embeddings(
                embeddings_file=args.embeddings_file,
                method="hmm",
                output_dir=args.output,
                n_states=args.states,
                covariance_type=args.covariance,
                pca_components=args.pca_components
            )
        else:
            # Clustering analysis
            from sequence_scribe.analysis import ClusteringAnalyzer
            analyzer = ClusteringAnalyzer(output_dir=args.output)
            results = analyzer.analyze(
                embeddings_file=args.embeddings_file,
                method="kmeans",
                n_clusters=args.states
            )
        
        print("\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {results.get('output_directory', args.output)}")
        
        if args.method == "hmm" and results.get('train_log_likelihood'):
            print(f"📊 Train log likelihood: {results['train_log_likelihood']:.2f}")
            print(f"📊 Test log likelihood: {results.get('test_log_likelihood', 'N/A'):.2f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error analyzing embeddings: {e}")
        return 1

def cmd_embed_text(args) -> int:
    """Handle the embed-text command."""
    try:
        print(f"Creating text embeddings from: {args.csv_path}")
        print(f"Model: {args.model}")
        
        results = ss.create_text_embeddings(
            csv_path=args.csv_path,
            output_dir=args.output,
            model_name=args.model
        )
        
        print("\n✅ Text embeddings created successfully!")
        print(f"📁 Results saved to: {results['output_directory']}")
        print(f"📊 Embedding shape: {results['embedding_shape']}")
        print(f"🎯 Embeddings file: {results['embeddings_file']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error creating text embeddings: {e}")
        return 1

def cmd_embed_graph(args) -> int:
    """Handle the embed-graph command."""
    try:
        print(f"Creating graph embeddings from: {args.csv_path}")
        print(f"Proximity threshold: {args.proximity_threshold}")
        
        results = ss.create_graph_embeddings(
            csv_path=args.csv_path,
            output_dir=args.output,
            proximity_threshold=args.proximity_threshold,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim
        )
        
        print("\n✅ Graph embeddings created successfully!")
        print(f"📁 Results saved to: {results['output_directory']}")
        print(f"📊 Embedding shape: {results['embedding_shape']}")
        print(f"🎯 Embeddings file: {results['embeddings_file']}")
        print(f"📈 Number of graphs: {results['num_graphs']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error creating graph embeddings: {e}")
        return 1

def cmd_compare(args) -> int:
    """Handle the compare command."""
    try:
        print(f"Comparing methods {args.methods} on: {args.video_path}")
        
        scribe = ss.SequenceScribe(output_dir=args.output)
        results = scribe.compare_methods(
            video_path=args.video_path,
            methods=args.methods
        )
        
        print("\n✅ Method comparison completed!")
        print(f"📁 Results saved to: {results.get('output_directory', args.output)}")
        print("📊 Comparison results:")
        
        for method in args.methods:
            if method in results["results_by_method"]:
                method_results = results["results_by_method"][method]
                print(f"  {method.upper()}:")
                if "embedding_shape" in method_results:
                    print(f"    - Embedding shape: {method_results['embedding_shape']}")
                if "analysis" in method_results:
                    analysis = method_results["analysis"]
                    if "train_log_likelihood" in analysis:
                        print(f"    - Train LL: {analysis['train_log_likelihood']:.2f}")
                    if "test_log_likelihood" in analysis:
                        print(f"    - Test LL: {analysis['test_log_likelihood']:.2f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error comparing methods: {e}")
        return 1

def cmd_status(args) -> int:
    """Handle the status command."""
    print("🔍 Checking Sequence Scribe system status...\n")
    
    # Check dependencies
    deps = check_dependencies()
    
    print("📦 Dependencies:")
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        print(f"  {status} {dep}")
    
    # Check GPU availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"\n🚀 GPU Support:")
        print(f"  {'✅' if cuda_available else '❌'} CUDA available: {cuda_available}")
        if cuda_available:
            print(f"  🎯 GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print(f"\n🚀 GPU Support:")
        print(f"  ❌ PyTorch not available")
    
    # Overall status
    critical_deps = ["torch", "numpy", "pandas", "sklearn", "matplotlib"]
    all_critical_available = all(deps.get(dep, False) for dep in critical_deps)
    
    print(f"\n🎯 Overall Status: {'✅ Ready' if all_critical_available else '⚠️  Some dependencies missing'}")
    
    if not all_critical_available:
        missing = [dep for dep in critical_deps if not deps.get(dep, False)]
        print(f"   Missing critical dependencies: {', '.join(missing)}")
        return 1
    
    return 0

def cmd_info(args) -> int:
    """Handle the info command."""
    print("📋 System Information\n")
    
    info = get_system_info()
    
    print(f"🐍 Python: {info['python_version'].split()[0]}")
    print(f"💻 Platform: {info['platform']}")
    print(f"🔧 Architecture: {' '.join(info['architecture'])}")
    
    if info.get('cuda_available'):
        print(f"🚀 CUDA: {info['cuda_version']}")
        print(f"🎯 GPUs: {info['gpu_count']}")
        for i, gpu_name in enumerate(info.get('gpu_names', [])):
            print(f"    GPU {i}: {gpu_name}")
    else:
        print(f"🚀 CUDA: Not available")
    
    print(f"\n📦 Dependencies:")
    for dep, available in info['dependencies'].items():
        status = "✅" if available else "❌"
        print(f"  {status} {dep}")
    
    return 0

def main():
    """Main CLI entry point."""
    parser = create_main_parser()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return 1
    
    args = parser.parse_args()
    
    if args.command == "process":
        return cmd_process(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "embed-text":
        return cmd_embed_text(args)
    elif args.command == "embed-graph":
        return cmd_embed_graph(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())