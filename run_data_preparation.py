#!/usr/bin/env python3
"""
Simple script to run the improved data preparation pipeline
"""

import yaml
import sys
from pathlib import Path
from improved_data_preparation import ImprovedDataPreparation

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten nested config for compatibility
    flattened_config = {
        'artifact_dir': config['artifact_dir'],
        'artifact_version': config['artifact_version'],
        'random_state': config['random_state'],
        'test_size': config['test_size'],
        'valid_size': config['valid_size'],
        'include_sentiment': config['features']['sentiment'],
        'include_pos': config['features']['pos'],
        'include_readability': config['features']['readability'],
        'include_lexical': config['features']['lexical'],
        'include_statistical': config['features']['statistical'],
        'tfidf_ngram_range': tuple(config['tfidf']['ngram_range']),
        'tfidf_min_df': config['tfidf']['min_df'],
        'tfidf_max_df': config['tfidf']['max_df'],
        'tfidf_max_features': config['tfidf']['max_features'],
        'tfidf_sublinear_tf': config['tfidf']['sublinear_tf'],
        'tfidf_norm': config['tfidf']['norm'],
        'vocab_size': config['sequences']['vocab_size'],
        'max_len': config['sequences']['max_len'],
        'max_len_percentile': config['sequences']['max_len_percentile'],
        'hf_model_name': config['transformer']['model_name'],
        'transformer_max_length': config['transformer']['max_length'],
        'enable_augmentation': config['augmentation']['enabled']
    }
    
    return flattened_config

def main():
    """Main function"""
    print("Starting improved data preparation pipeline...")
    
    # Load configuration
    try:
        config = load_config()
        print("Configuration loaded successfully")
    except FileNotFoundError:
        print("Error: config.yaml not found. Using default configuration.")
        config = {
            'artifact_dir': 'artifacts',
            'artifact_version': 'v2',
            'random_state': 42,
            'test_size': 0.10,
            'valid_size': 0.10,
            'include_sentiment': True,
            'include_pos': True,
            'include_readability': True,
            'include_lexical': True,
            'include_statistical': True,
            'tfidf_ngram_range': (1, 2),
            'tfidf_min_df': 5,
            'tfidf_max_df': 0.8,
            'tfidf_max_features': 200000,
            'tfidf_sublinear_tf': True,
            'tfidf_norm': 'l2',
            'vocab_size': 40000,
            'max_len': None,
            'max_len_percentile': 95,
            'hf_model_name': 'distilbert-base-uncased',
            'transformer_max_length': 128,
            'enable_augmentation': True
        }
    
    # Initialize pipeline
    pipeline = ImprovedDataPreparation(config)
    
    # Define dataset paths - UPDATE THESE WITH YOUR ACTUAL FILE PATHS
    dataset_paths = {
        'dailydialog': 'daily_dialog.csv',
        'isear': 'ISEAR.csv',
        'goemotions_1': 'goemotions_1.csv',
        'goemotions_2': 'goemotions_2.csv',
        'goemotions_3': 'goemotions_3.csv'
    }
    
    # Check if dataset files exist
    missing_files = []
    for name, path in dataset_paths.items():
        if not Path(path).exists():
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        print("\nWarning: The following dataset files are missing:")
        for missing in missing_files:
            print(f"  - {missing}")
        print("\nPlease ensure all dataset files are in the current directory.")
        print("You can also modify the dataset_paths dictionary in this script.")
        
        # Ask user if they want to continue with available files
        response = input("\nDo you want to continue with available files only? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(1)
        
        # Filter out missing files
        dataset_paths = {k: v for k, v in dataset_paths.items() if Path(v).exists()}
    
    if not dataset_paths:
        print("Error: No dataset files found. Please check your file paths.")
        sys.exit(1)
    
    try:
        # Run the pipeline
        print(f"\nProcessing {len(dataset_paths)} datasets...")
        
        # Load and validate data
        df = pipeline.load_and_validate_data(dataset_paths)
        print(f"✓ Loaded {len(df)} samples")
        
        # Analyze data quality
        quality_analysis = pipeline.analyze_data_quality(df)
        print(f"✓ Data quality analysis complete")
        print(f"  - Class balance ratio: {quality_analysis['class_balance']['balance_ratio']:.3f}")
        print(f"  - Quality issues: {len(quality_analysis['quality_issues'])}")
        
        # Apply data augmentation if needed
        if quality_analysis['class_balance']['balance_ratio'] < 0.3:
            print("✓ Applying data augmentation for class balance...")
            df = pipeline.apply_data_augmentation(df)
            print(f"  - New sample count: {len(df)}")
        
        # Create stratified splits
        print("✓ Creating stratified train/validation/test splits...")
        splits = pipeline.create_stratified_splits(df)
        
        # Extract features
        print("✓ Extracting features...")
        features = pipeline.extract_enhanced_features(df)
        
        # Save everything
        print("✓ Saving processed data and features...")
        pipeline.save_processed_data(df, features)
        
        print("\n🎉 Data preparation completed successfully!")
        print(f"📁 Artifacts saved to: {config['artifact_dir']}")
        print(f"📊 Final dataset size: {len(df)} samples")
        print(f"🏷️  Classes: {df['label'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"\n❌ Error during data preparation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()