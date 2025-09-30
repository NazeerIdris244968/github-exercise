#!/usr/bin/env python3
"""
Example usage of the improved data preparation pipeline
for emotion classification task.

This script demonstrates how to use the pipeline with your specific datasets
and shows the key improvements over the original approach.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from improved_data_preparation import ImprovedDataPreparation

def demonstrate_improvements():
    """Demonstrate key improvements of the new pipeline"""
    
    print("🚀 Improved Data Preparation Pipeline Demo")
    print("=" * 50)
    
    # Configuration for the emotion classification task
    config = {
        'artifact_dir': 'artifacts',
        'artifact_version': 'v2',
        'random_state': 42,
        
        # Data splitting (same as original)
        'test_size': 0.10,
        'valid_size': 0.10,
        
        # Enhanced feature extraction
        'include_sentiment': True,
        'include_pos': True,
        'include_readability': True,
        'include_lexical': True,
        'include_statistical': True,
        
        # TF-IDF parameters (improved)
        'tfidf_ngram_range': (1, 3),  # Extended to trigrams
        'tfidf_min_df': 3,  # More aggressive filtering
        'tfidf_max_df': 0.7,  # More aggressive filtering
        'tfidf_max_features': 100000,  # Reasonable size
        'tfidf_sublinear_tf': True,
        'tfidf_norm': 'l2',
        
        # Sequence parameters
        'vocab_size': 30000,
        'max_len': None,
        'max_len_percentile': 95,
        
        # Transformer parameters
        'hf_model_name': 'distilbert-base-uncased',
        'transformer_max_length': 128,
        
        # Data augmentation
        'enable_augmentation': True,
    }
    
    # Initialize pipeline
    pipeline = ImprovedDataPreparation(config)
    
    # Simulate dataset loading (replace with your actual file paths)
    print("\n📊 Dataset Loading and Validation")
    print("-" * 30)
    
    # Example dataset paths - UPDATE THESE WITH YOUR ACTUAL PATHS
    dataset_paths = {
        'dailydialog': 'daily_dialog.csv',
        'isear': 'ISEAR.csv',
        'goemotions_1': 'goemotions_1.csv',
        'goemotions_2': 'goemotions_2.csv',
        'goemotions_3': 'goemotions_3.csv'
    }
    
    # Check if files exist
    available_datasets = {}
    for name, path in dataset_paths.items():
        if Path(path).exists():
            available_datasets[name] = path
            print(f"✓ Found: {name}")
        else:
            print(f"✗ Missing: {name} ({path})")
    
    if not available_datasets:
        print("\n⚠️  No dataset files found. Creating sample data for demonstration...")
        df = create_sample_data()
    else:
        print(f"\n📁 Loading {len(available_datasets)} datasets...")
        df = pipeline.load_and_validate_data(available_datasets)
    
    print(f"✓ Loaded {len(df)} samples")
    
    # Data quality analysis
    print("\n🔍 Data Quality Analysis")
    print("-" * 25)
    
    quality_analysis = pipeline.analyze_data_quality(df)
    
    print(f"Total samples: {quality_analysis['total_samples']}")
    print(f"Unique texts: {quality_analysis['unique_texts']}")
    print(f"Duplicate ratio: {quality_analysis['duplicate_ratio']:.3f}")
    print(f"Class balance ratio: {quality_analysis['class_balance']['balance_ratio']:.3f}")
    
    if quality_analysis['quality_issues']:
        print("Quality issues found:")
        for issue in quality_analysis['quality_issues']:
            print(f"  - {issue}")
    else:
        print("✓ No quality issues detected")
    
    # Class distribution
    print("\n📈 Class Distribution:")
    for emotion, count in quality_analysis['class_distribution'].items():
        percentage = (count / quality_analysis['total_samples']) * 100
        print(f"  {emotion}: {count:,} ({percentage:.1f}%)")
    
    # Apply data augmentation if needed
    if quality_analysis['class_balance']['balance_ratio'] < 0.3:
        print("\n🔄 Applying Data Augmentation")
        print("-" * 28)
        original_size = len(df)
        df = pipeline.apply_data_augmentation(df)
        print(f"✓ Augmented: {original_size} → {len(df)} samples")
    else:
        print("\n✓ Class balance is acceptable, skipping augmentation")
    
    # Create stratified splits
    print("\n✂️  Creating Stratified Splits")
    print("-" * 28)
    
    splits = pipeline.create_stratified_splits(df)
    
    for split_name, split_df in splits.items():
        print(f"{split_name.capitalize()}: {len(split_df)} samples")
        class_dist = split_df['label'].value_counts().to_dict()
        print(f"  Classes: {class_dist}")
    
    # Extract features
    print("\n🔧 Feature Extraction")
    print("-" * 20)
    
    features = pipeline.extract_enhanced_features(df)
    
    print("Extracted feature types:")
    for feature_type, feature_data in features.items():
        if isinstance(feature_data, pd.DataFrame):
            print(f"  ✓ {feature_type}: {feature_data.shape[1]} features")
        elif isinstance(feature_data, dict):
            print(f"  ✓ {feature_type}: {len(feature_data)} components")
        else:
            print(f"  ✓ {feature_type}: {type(feature_data).__name__}")
    
    # Show feature details
    print("\n📋 Feature Details:")
    
    if 'sentiment' in features and not features['sentiment'].empty:
        print(f"  Sentiment features: {list(features['sentiment'].columns)}")
    
    if 'pos' in features and not features['pos'].empty:
        pos_cols = [col for col in features['pos'].columns if col.endswith('_ratio')]
        print(f"  POS features: {len(pos_cols)} normalized ratios")
    
    if 'lexical' in features and not features['lexical'].empty:
        print(f"  Lexical features: {list(features['lexical'].columns)}")
    
    # Save everything
    print("\n💾 Saving Processed Data")
    print("-" * 22)
    
    pipeline.save_processed_data(df, features)
    
    print("✓ All data and features saved successfully!")
    print(f"📁 Artifacts directory: {config['artifact_dir']}")
    
    # Show final statistics
    print("\n📊 Final Statistics")
    print("-" * 18)
    
    print(f"Total samples: {len(df):,}")
    print(f"Feature types: {len(features)}")
    
    # Count total features
    total_features = 0
    for feature_type, feature_data in features.items():
        if isinstance(feature_data, pd.DataFrame):
            total_features += feature_data.shape[1]
        elif isinstance(feature_data, dict) and 'train' in feature_data:
            # TF-IDF or sequence features
            if hasattr(feature_data['train'], 'shape'):
                total_features += feature_data['train'].shape[1]
    
    print(f"Total features: {total_features:,}")
    
    # Show artifact structure
    print("\n📁 Generated Artifacts:")
    show_artifact_structure(config['artifact_dir'])
    
    print("\n🎉 Data preparation completed successfully!")
    print("\nNext steps:")
    print("1. Use the generated features to train your models")
    print("2. Check the quality reports in artifacts/reports/")
    print("3. Use the stratified splits for consistent evaluation")

def create_sample_data():
    """Create sample data for demonstration if real datasets are not available"""
    print("Creating sample emotion data for demonstration...")
    
    sample_texts = [
        "I'm so happy today! This is amazing!",
        "I feel really sad about what happened.",
        "Wow, that's surprising! I didn't expect that.",
        "I'm so angry right now, this is ridiculous!",
        "That's disgusting, I can't believe it.",
        "I'm scared about what might happen next.",
        "I don't feel anything particular about this.",
        "This is wonderful news! I'm thrilled!",
        "I'm disappointed with the results.",
        "What a shock! I'm completely surprised!",
        "I'm furious about this situation!",
        "That's repulsive, I can't stand it.",
        "I'm terrified of what's coming.",
        "I'm neutral about this topic.",
        "I love this so much! It's fantastic!",
        "I'm heartbroken by this news.",
        "I can't believe my eyes! This is incredible!",
        "I'm outraged by this behavior!",
        "That's absolutely revolting!",
        "I'm anxious about the future.",
    ]
    
    sample_labels = [
        "happiness", "sadness", "surprise", "anger", "disgust",
        "fear", "neutral", "happiness", "sadness", "surprise",
        "anger", "disgust", "fear", "neutral", "happiness",
        "sadness", "surprise", "anger", "disgust", "fear"
    ]
    
    # Create more samples by repeating and adding variations
    expanded_texts = []
    expanded_labels = []
    
    for i in range(50):  # Create 1000 samples
        text = sample_texts[i % len(sample_texts)]
        label = sample_labels[i % len(sample_labels)]
        
        # Add some variation
        if i % 3 == 0:
            text = text.upper()
        elif i % 3 == 1:
            text = text.lower()
        
        expanded_texts.append(text)
        expanded_labels.append(label)
    
    return pd.DataFrame({
        'text': expanded_texts,
        'label': expanded_labels,
        'source': 'sample'
    })

def show_artifact_structure(artifact_dir):
    """Show the structure of generated artifacts"""
    artifact_path = Path(artifact_dir)
    
    if not artifact_path.exists():
        print("  No artifacts directory found")
        return
    
    def print_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        items = sorted(path.iterdir())
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and current_depth < max_depth - 1:
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item, next_prefix, max_depth, current_depth + 1)
    
    print_tree(artifact_path)

def compare_with_original():
    """Compare the improved pipeline with the original approach"""
    
    print("\n🆚 Comparison with Original Pipeline")
    print("=" * 40)
    
    improvements = [
        "✅ Enhanced data validation with quality metrics",
        "✅ Advanced text preprocessing and cleaning",
        "✅ Comprehensive feature extraction (8 types vs 3)",
        "✅ Automatic data augmentation for class balance",
        "✅ Robust stratified splitting with validation",
        "✅ Detailed logging and progress tracking",
        "✅ YAML-based configuration system",
        "✅ Artifact versioning and organization",
        "✅ Error handling and recovery",
        "✅ Quality reports and analysis",
        "✅ Support for multiple model architectures",
        "✅ Memory-efficient batch processing"
    ]
    
    print("Key improvements:")
    for improvement in improvements:
        print(f"  {improvement}")
    
    print("\nExpected benefits:")
    print("  🎯 Better model performance through higher quality data")
    print("  🎯 Richer feature set for different model types")
    print("  🎯 Improved class balance through augmentation")
    print("  🎯 Better reproducibility and debugging")
    print("  🎯 Easier integration with different model architectures")

if __name__ == "__main__":
    try:
        demonstrate_improvements()
        compare_with_original()
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()