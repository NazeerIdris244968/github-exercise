# Improved Data Preparation Pipeline for Emotion Classification

This repository contains an enhanced data preparation pipeline for emotion classification tasks, specifically designed for the Task 6: Model Iterations assignment.

## 🚀 Key Improvements

### 1. **Enhanced Data Validation & Quality Checks**
- Comprehensive data quality analysis with detailed reporting
- Automatic detection of data issues (duplicates, missing values, class imbalance)
- Quality metrics tracking and logging
- Robust error handling and validation

### 2. **Advanced Text Preprocessing**
- Multi-level text cleaning pipeline
- Invisible character removal
- Punctuation normalization
- Length-based filtering with configurable thresholds
- Dataset-specific cleaning logic for DailyDialog, ISEAR, and GoEmotions

### 3. **Comprehensive Feature Extraction**
- **Sentiment Features**: VADER sentiment + additional metrics (positive/negative word counts, ratios)
- **POS Features**: Complete Universal POS tag analysis with normalized ratios
- **Readability Features**: Flesch-Kincaid grade, sentence length analysis
- **Lexical Features**: Vocabulary richness, stop word ratios, punctuation analysis
- **Statistical Features**: Word length statistics, syllable counting
- **TF-IDF Features**: Configurable n-gram ranges and parameters
- **Sequence Features**: Tokenization and padding for deep learning models
- **Transformer Features**: HuggingFace tokenizer integration

### 4. **Improved Data Splitting**
- Stratified splitting ensuring class balance across all splits
- Configurable train/validation/test ratios
- Reproducible splits with random state control
- Split validation and reporting

### 5. **Data Augmentation**
- Automatic class balance detection
- Synonym-based augmentation using WordNet
- Configurable augmentation thresholds
- Source tracking for augmented samples

### 6. **Robust Configuration & Logging**
- YAML-based configuration system
- Comprehensive logging with file and console output
- Progress tracking and error reporting
- Artifact versioning and organization

## 📁 File Structure

```
workspace/
├── improved_data_preparation.py    # Main pipeline implementation
├── run_data_preparation.py         # Simple execution script
├── config.yaml                     # Configuration file
├── requirements.txt                # Python dependencies
├── README_data_preparation.md      # This file
└── artifacts/                      # Generated artifacts
    ├── splits/                     # Train/valid/test indices
    ├── labels/                     # Label mappings and targets
    ├── features/                   # Extracted features
    │   ├── dense/                  # Dense features (sentiment, POS, etc.)
    │   ├── tfidf/                  # TF-IDF matrices
    │   ├── sequences/              # Tokenized sequences
    │   └── transformers/           # Transformer tokenization
    ├── augmented/                  # Augmented data
    └── reports/                    # Quality analysis reports
```

## 🛠️ Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download spaCy model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Download NLTK data:**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```

## 🚀 Usage

### Quick Start

1. **Place your dataset files in the workspace directory:**
   - `daily_dialog.csv`
   - `ISEAR.csv`
   - `goemotions_1.csv`
   - `goemotions_2.csv`
   - `goemotions_3.csv`

2. **Run the pipeline:**
   ```bash
   python run_data_preparation.py
   ```

### Advanced Usage

```python
from improved_data_preparation import ImprovedDataPreparation

# Load configuration
config = {
    'artifact_dir': 'artifacts',
    'artifact_version': 'v2',
    'test_size': 0.10,
    'valid_size': 0.10,
    'include_sentiment': True,
    'include_pos': True,
    # ... other parameters
}

# Initialize pipeline
pipeline = ImprovedDataPreparation(config)

# Define dataset paths
dataset_paths = {
    'dailydialog': 'daily_dialog.csv',
    'isear': 'ISEAR.csv',
    'goemotions_1': 'goemotions_1.csv',
    'goemotions_2': 'goemotions_2.csv',
    'goemotions_3': 'goemotions_3.csv'
}

# Run pipeline
df = pipeline.load_and_validate_data(dataset_paths)
quality_analysis = pipeline.analyze_data_quality(df)
splits = pipeline.create_stratified_splits(df)
features = pipeline.extract_enhanced_features(df)
pipeline.save_processed_data(df, features)
```

## ⚙️ Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
# Data splitting
test_size: 0.10
valid_size: 0.10

# Feature extraction
features:
  sentiment: true
  pos: true
  readability: true
  lexical: true
  statistical: true

# TF-IDF parameters
tfidf:
  ngram_range: [1, 2]
  max_features: 200000

# Data augmentation
augmentation:
  enabled: true
  target_balance_ratio: 0.3
```

## 📊 Output

The pipeline generates:

1. **Processed Dataset**: Clean, validated data with consistent labels
2. **Feature Matrices**: TF-IDF, dense features, sequences, transformer tokens
3. **Data Splits**: Stratified train/validation/test indices
4. **Quality Reports**: Detailed analysis of data quality and class balance
5. **Label Mappings**: Consistent emotion label to ID mappings
6. **Scalers**: Fitted scalers for feature normalization

## 🔍 Key Features for Model Training

### Traditional Models (Logistic Regression, Naive Bayes, SVM)
- **TF-IDF features**: High-dimensional sparse features
- **Dense features**: Sentiment, POS, readability, lexical, statistical features
- **Scaled features**: Normalized dense features using MaxAbsScaler

### Deep Learning Models (LSTM, RNN)
- **Sequence features**: Tokenized and padded text sequences
- **Tokenizer**: Fitted on training data with configurable vocabulary size
- **Metadata**: Vocabulary size and sequence length information

### Transformer Models (BERT, DistilBERT, etc.)
- **Input IDs**: Tokenized text as integer sequences
- **Attention Masks**: Binary masks for padding tokens
- **Model Info**: HuggingFace model name for reproducibility

## 🎯 Benefits for Your Assignment

1. **Better Data Quality**: Comprehensive validation ensures clean, consistent data
2. **Rich Feature Set**: Multiple feature types for different model architectures
3. **Class Balance**: Automatic augmentation helps with imbalanced datasets
4. **Reproducibility**: Fixed random seeds and versioned artifacts
5. **Scalability**: Handles large datasets efficiently with batch processing
6. **Monitoring**: Detailed logging and quality reports for debugging

## 🐛 Troubleshooting

### Common Issues

1. **Missing spaCy model**: Run `python -m spacy download en_core_web_sm`
2. **NLTK data missing**: The script will download automatically on first run
3. **Memory issues**: Reduce `tfidf_max_features` or `vocab_size` in config
4. **File not found**: Ensure dataset files are in the correct directory

### Performance Tips

1. **Large datasets**: Use batch processing (already implemented)
2. **Memory optimization**: Process features separately if needed
3. **Parallel processing**: spaCy uses all available cores by default

## 📈 Expected Improvements

Compared to the original pipeline, you should see:

- **Better class balance** through automatic augmentation
- **Richer feature set** with 8 different feature types
- **Higher data quality** through comprehensive validation
- **Better reproducibility** with detailed logging and versioning
- **More robust splitting** with proper stratification
- **Easier debugging** with quality reports and progress tracking

This improved pipeline should help you achieve better model performance and meet the F1-score target of 0.75 for your emotion classification task.