def load_gnn_embeddings(npz_file: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load GNN embeddings from NPZ file with extensive debugging information.
    """
    try:
        # Load the NPZ file
        data = np.load(npz_file, allow_pickle=True)
        
        print(f"\n=== DEBUGGING NPZ FILE: {os.path.basename(npz_file)} ===")
        print(f"Total keys in NPZ: {len(data.files)}")
        print(f"NPZ file keys (first 20): {list(data.files)[:20]}")
        
        # Extract embeddings and image names
        embeddings_dict = {}
        for key in data.files:
            embeddings_dict[key] = data[key]
        
        # Check if keys look like frame names
        sample_keys = list(data.files)[:10]
        print(f"Sample keys analysis:")
        for key in sample_keys:
            print(f"  Key: '{key}' (type: {type(key)})")
            # Try to extract frame number
            try:
                if 'frame_' in str(key):
                    frame_num = str(key).split('frame_')[1].split('.')[0]
                    print(f"    -> Extracted frame number: {frame_num}")
                else:
                    print(f"    -> No 'frame_' pattern found")
            except:
                print(f"    -> Could not extract frame number")
        
        # Load metadata file to get proper ordering of images
        metadata_file = os.path.join(os.path.dirname(npz_file), "embeddings_metadata.json")
        if os.path.exists(metadata_file):
            print(f"\nFound metadata file: {metadata_file}")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print(f"Metadata keys: {list(metadata.keys())}")
            
            if "image_names" in metadata:
                image_names = metadata["image_names"]
                print(f"Metadata contains {len(image_names)} image names")
                print(f"First 10 image names from metadata: {image_names[:10]}")
                print(f"Last 10 image names from metadata: {image_names[-10:]}")
                
                # Check if metadata names match NPZ keys
                metadata_set = set(str(name) for name in image_names)
                npz_set = set(str(key) for key in data.files)
                
                missing_in_npz = metadata_set - npz_set
                missing_in_metadata = npz_set - metadata_set
                
                print(f"\nCross-reference check:")
                print(f"  Names in metadata but not in NPZ: {len(missing_in_npz)}")
                if len(missing_in_npz) > 0:
                    print(f"    Examples: {list(missing_in_npz)[:5]}")
                
                print(f"  Names in NPZ but not in metadata: {len(missing_in_metadata)}")
                if len(missing_in_metadata) > 0:
                    print(f"    Examples: {list(missing_in_metadata)[:5]}")
                
                # Check frame number patterns in metadata
                try:
                    frame_numbers = []
                    for name in image_names[:20]:  # Check first 20
                        if 'frame_' in str(name):
                            frame_num = int(str(name).split('frame_')[1].split('.')[0])
                            frame_numbers.append(frame_num)
                    
                    if frame_numbers:
                        print(f"\nFrame number analysis from metadata (first 20):")
                        print(f"  Range: {min(frame_numbers)} to {max(frame_numbers)}")
                        print(f"  Sorted sample: {sorted(frame_numbers)[:10]}")
                        print(f"  Sequential? {frame_numbers == list(range(min(frame_numbers), max(frame_numbers)+1))}")
                except Exception as e:
                    print(f"  Could not analyze frame numbers: {e}")
                
            else:
                print(f"No 'image_names' key in metadata")
                image_names = list(embeddings_dict.keys())
        else:
            print(f"No metadata file found at {metadata_file}")
            print("Using NPZ keys directly as image names")
            image_names = list(embeddings_dict.keys())
        
        # Ensure image_names are strings
        image_names = [str(name) for name in image_names]
        
        # Final frame number analysis on the actual image_names we'll use
        print(f"\n=== FINAL IMAGE_NAMES ANALYSIS ===")
        print(f"Total image names: {len(image_names)}")
        print(f"First 10: {image_names[:10]}")
        print(f"Last 10: {image_names[-10:]}")
        
        # Try to extract and analyze frame numbers from final image_names
        try:
            frame_numbers = []
            for name in image_names:
                if 'frame_' in name:
                    frame_num = int(name.split('frame_')[1].split('.')[0])
                    frame_numbers.append(frame_num)
            
            if frame_numbers:
                print(f"\nFrame number statistics:")
                print(f"  Total frames with numbers: {len(frame_numbers)}")
                print(f"  Range: {min(frame_numbers)} to {max(frame_numbers)}")
                print(f"  Unique frames: {len(set(frame_numbers))}")
                print(f"  Duplicates: {len(frame_numbers) - len(set(frame_numbers))}")
                
                # Check gaps in sequence
                sorted_nums = sorted(set(frame_numbers))
                gaps = []
                for i in range(len(sorted_nums) - 1):
                    gap = sorted_nums[i+1] - sorted_nums[i]
                    if gap > 1:
                        gaps.append((sorted_nums[i], sorted_nums[i+1], gap))
                
                print(f"  Sequence gaps (>1): {len(gaps)}")
                if gaps:
                    print(f"    First 5 gaps: {gaps[:5]}")
                
                # Show distribution
                print(f"  First 20 frame numbers: {frame_numbers[:20]}")
                print(f"  Are they consecutive? {frame_numbers == list(range(frame_numbers[0], frame_numbers[0] + len(frame_numbers)))}")
                
        except Exception as e:
            print(f"Could not analyze frame numbers: {e}")
        
        # Check embedding dimensions
        sample_embedding = next(iter(embeddings_dict.values()))
        print(f"\nEmbedding info:")
        print(f"  Embedding dimension: {sample_embedding.shape}")
        print(f"  Embedding dtype: {sample_embedding.dtype}")
        print(f"  Sample embedding preview: {sample_embedding[:5] if len(sample_embedding) > 5 else sample_embedding}")
        
        print(f"=== END DEBUGGING ===\n")
        
        return embeddings_dict, image_names
    
    except Exception as e:
        print(f"Error loading GNN embeddings from {npz_file}: {e}")
        import traceback
        traceback.print_exc()
        raise
