#!/usr/bin/env python3
"""
Improved Data Preparation Pipeline for Emotion Classification Task

This script provides an enhanced data preparation pipeline that addresses:
1. Better data validation and quality checks
2. Advanced text preprocessing and cleaning
3. Improved class balance handling
4. Enhanced feature extraction
5. Robust data splitting with stratification
6. Data augmentation techniques
7. Comprehensive logging and monitoring

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Data science libraries
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
import joblib
from scipy import sparse

# NLP libraries
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease, flesch_kincaid_grade
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Deep learning libraries
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer

# Data augmentation
from nlpaug import Augmenter
from nlpaug.augmenter.word import SynonymAug, RandomWordAug
from nlpaug.augmenter.sentence import BackTranslationAug

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImprovedDataPreparation:
    """
    Enhanced data preparation pipeline for emotion classification
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the data preparation pipeline
        
        Args:
            config: Configuration dictionary containing all parameters
        """
        self.config = config
        self.artifact_dir = Path(config['artifact_dir'])
        self.artifact_version = config['artifact_version']
        
        # Create directories
        self._create_directories()
        
        # Initialize components
        self._initialize_components()
        
        # Data storage
        self.data = {}
        self.features = {}
        self.splits = {}
        
    def _create_directories(self):
        """Create necessary directories for artifacts"""
        dirs = [
            self.artifact_dir,
            self.artifact_dir / "splits",
            self.artifact_dir / "labels", 
            self.artifact_dir / "features" / "dense",
            self.artifact_dir / "features" / "tfidf",
            self.artifact_dir / "features" / "sequences",
            self.artifact_dir / "features" / "transformers",
            self.artifact_dir / "augmented",
            self.artifact_dir / "reports"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _initialize_components(self):
        """Initialize NLP components and models"""
        try:
            # Initialize spaCy model
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model not found. Please install: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        # Initialize VADER sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize NLTK components
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            logger.info("NLTK components initialized")
        except Exception as e:
            logger.warning(f"NLTK initialization failed: {e}")
            self.stop_words = set()
            self.lemmatizer = None
            
        # Initialize data augmentation
        if self.config.get('enable_augmentation', False):
            try:
                self.augmenter = SynonymAug(aug_src='wordnet')
                logger.info("Data augmentation enabled")
            except Exception as e:
                logger.warning(f"Data augmentation initialization failed: {e}")
                self.augmenter = None
        else:
            self.augmenter = None

    def load_and_validate_data(self, file_paths: Dict[str, str]) -> pd.DataFrame:
        """
        Load and validate datasets with comprehensive quality checks
        
        Args:
            file_paths: Dictionary mapping dataset names to file paths
            
        Returns:
            Combined and validated DataFrame
        """
        logger.info("Starting data loading and validation")
        
        datasets = {}
        
        for name, path in file_paths.items():
            logger.info(f"Loading {name} from {path}")
            
            try:
                # Load dataset
                if path.endswith('.tsv'):
                    df = pd.read_csv(path, sep='\t')
                elif path.endswith('.csv'):
                    df = pd.read_csv(path)
                else:
                    raise ValueError(f"Unsupported file format: {path}")
                
                # Basic validation
                if 'text' not in df.columns or 'label' not in df.columns:
                    raise ValueError(f"Required columns 'text' and 'label' not found in {name}")
                
                # Clean and validate
                df = self._clean_dataset(df, name)
                datasets[name] = df
                
                logger.info(f"{name} loaded: {len(df)} samples")
                
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
                continue
        
        if not datasets:
            raise ValueError("No datasets loaded successfully")
        
        # Combine datasets
        combined_df = pd.concat(datasets.values(), ignore_index=True)
        
        # Final validation and cleaning
        combined_df = self._final_validation(combined_df)
        
        self.data['raw'] = combined_df
        logger.info(f"Combined dataset: {len(combined_df)} samples")
        
        return combined_df
    
    def _clean_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Clean individual dataset with dataset-specific logic
        
        Args:
            df: Input DataFrame
            dataset_name: Name of the dataset for specific processing
            
        Returns:
            Cleaned DataFrame
        """
        # Remove duplicates
        initial_size = len(df)
        df = df.drop_duplicates(subset=['text', 'label'])
        logger.info(f"{dataset_name}: Removed {initial_size - len(df)} duplicates")
        
        # Clean text column
        df['text'] = df['text'].astype(str).str.strip()
        df = df[df['text'].str.len() > 0]
        
        # Dataset-specific cleaning
        if dataset_name == 'dailydialog':
            df = self._clean_dailydialog(df)
        elif dataset_name == 'isear':
            df = self._clean_isear(df)
        elif dataset_name == 'goemotions':
            df = self._clean_goemotions(df)
        
        # Add source column
        df['source'] = dataset_name
        
        return df
    
    def _clean_dailydialog(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DailyDialog dataset"""
        import ast
        
        # Parse dialog and emotion columns
        df['dialog'] = df['dialog'].apply(
            lambda x: ast.literal_eval(str(x)) if str(x).startswith("[") else [str(x)]
        )
        
        df['emotion'] = df['emotion'].apply(
            lambda x: [int(n) for n in re.split(r"[ ,]+", str(x).strip("[]")) if n.isdigit()]
        )
        
        # Map emotions
        emotion_map = {
            0: "neutral", 1: "anger", 2: "disgust", 3: "fear",
            4: "happiness", 5: "sadness", 6: "surprise"
        }
        
        # Extract text-label pairs
        rows = []
        for _, row in df.iterrows():
            dialogs = row['dialog']
            emotions = row['emotion']
            
            min_len = min(len(dialogs), len(emotions))
            for i in range(min_len):
                text = str(dialogs[i]).strip()
                emotion_id = emotions[i]
                emotion_label = emotion_map.get(emotion_id)
                
                if text and emotion_label:
                    rows.append({
                        'text': text,
                        'label': emotion_label
                    })
        
        return pd.DataFrame(rows)
    
    def _clean_isear(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean ISEAR dataset"""
        # Map emotions
        emotion_map = {
            "anger": "anger",
            "disgust": "disgust", 
            "fear": "fear",
            "joy": "happiness",
            "happiness": "happiness",
            "sadness": "sadness",
            "shame": "sadness",
            "guilt": "sadness",
            "neutral": "neutral"
        }
        
        # Normalize sentiment
        df['sentiment_norm'] = df['sentiment'].astype(str).str.lower().str.strip()
        df['label'] = df['sentiment_norm'].map(emotion_map)
        
        # Clean text
        df['text'] = df['content'].astype(str).str.strip()
        
        # Remove invalid entries
        df = df.dropna(subset=['label'])
        df = df[df['text'].str.len() > 0]
        
        return df[['text', 'label']]
    
    def _clean_goemotions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean GoEmotions dataset"""
        # Define emotion mapping
        emotion_map = {
            # happiness
            "admiration": "happiness", "amusement": "happiness", "approval": "happiness",
            "caring": "happiness", "desire": "happiness", "excitement": "happiness",
            "gratitude": "happiness", "joy": "happiness", "love": "happiness",
            "optimism": "happiness", "pride": "happiness", "relief": "happiness",
            
            # anger
            "anger": "anger", "annoyance": "anger", "disapproval": "anger",
            
            # disgust
            "disgust": "disgust",
            
            # fear
            "fear": "fear", "nervousness": "fear",
            
            # sadness
            "sadness": "sadness", "disappointment": "sadness", "grief": "sadness",
            "remorse": "sadness", "embarrassment": "sadness",
            
            # surprise
            "surprise": "surprise", "realization": "surprise", "confusion": "surprise",
            "curiosity": "surprise",
            
            # neutral
            "neutral": "neutral"
        }
        
        # Get emotion columns (exclude metadata)
        meta_cols = {'text', 'id', 'author', 'subreddit', 'link_id', 'parent_id', 
                    'created_utc', 'rater_id', 'example_very_unclear'}
        emotion_cols = [col for col in df.columns if col not in meta_cols]
        
        # Process each row
        rows = []
        for _, row in df.iterrows():
            text = str(row['text']).strip()
            if not text:
                continue
                
            # Find active emotions
            active_emotions = []
            for col in emotion_cols:
                if col in emotion_map and row[col] == 1:
                    mapped_emotion = emotion_map[col]
                    active_emotions.append(mapped_emotion)
            
            # Keep only single-label entries
            unique_emotions = list(set(active_emotions))
            if len(unique_emotions) == 1:
                rows.append({
                    'text': text,
                    'label': unique_emotions[0]
                })
        
        return pd.DataFrame(rows)
    
    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final validation and cleaning of combined dataset
        
        Args:
            df: Combined DataFrame
            
        Returns:
            Validated and cleaned DataFrame
        """
        logger.info("Performing final validation")
        
        initial_size = len(df)
        
        # Remove rows with missing text or labels
        df = df.dropna(subset=['text', 'label'])
        
        # Remove empty text
        df = df[df['text'].str.len() > 0]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['text', 'label'])
        
        # Filter to target emotions only
        target_emotions = {'happiness', 'sadness', 'surprise', 'anger', 'disgust', 'fear', 'neutral'}
        df = df[df['label'].isin(target_emotions)]
        
        # Advanced text cleaning
        df['text'] = self._advanced_text_cleaning(df['text'])
        
        # Remove very short or very long texts
        text_lengths = df['text'].str.len()
        df = df[(text_lengths >= 3) & (text_lengths <= 1000)]
        
        # Shuffle
        df = df.sample(frac=1.0, random_state=self.config['random_state']).reset_index(drop=True)
        
        logger.info(f"Final validation: {initial_size} -> {len(df)} samples")
        
        return df
    
    def _advanced_text_cleaning(self, text_series: pd.Series) -> pd.Series:
        """
        Advanced text cleaning pipeline
        
        Args:
            text_series: Series of text strings
            
        Returns:
            Cleaned text series
        """
        def clean_text(text):
            # Convert to string and strip
            text = str(text).strip()
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove invisible characters
            text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
            
            # Remove excessive punctuation
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
            text = re.sub(r'[.]{2,}', '.', text)
            
            # Normalize quotes
            text = re.sub(r'["""]', '"', text)
            text = re.sub(r'[''']', "'", text)
            
            return text.strip()
        
        return text_series.apply(clean_text)
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive data quality analysis
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing quality metrics
        """
        logger.info("Analyzing data quality")
        
        analysis = {}
        
        # Basic statistics
        analysis['total_samples'] = len(df)
        analysis['unique_texts'] = df['text'].nunique()
        analysis['duplicate_ratio'] = 1 - (df['text'].nunique() / len(df))
        
        # Class distribution
        class_counts = df['label'].value_counts()
        analysis['class_distribution'] = class_counts.to_dict()
        analysis['class_balance'] = {
            'min_count': class_counts.min(),
            'max_count': class_counts.max(),
            'balance_ratio': class_counts.min() / class_counts.max()
        }
        
        # Text length statistics
        text_lengths = df['text'].str.len()
        analysis['text_length'] = {
            'mean': text_lengths.mean(),
            'std': text_lengths.std(),
            'min': text_lengths.min(),
            'max': text_lengths.max(),
            'median': text_lengths.median()
        }
        
        # Source distribution
        if 'source' in df.columns:
            analysis['source_distribution'] = df['source'].value_counts().to_dict()
        
        # Quality issues
        issues = []
        
        # Check for very short texts
        short_texts = (text_lengths < 3).sum()
        if short_texts > 0:
            issues.append(f"{short_texts} texts shorter than 3 characters")
        
        # Check for very long texts
        long_texts = (text_lengths > 1000).sum()
        if long_texts > 0:
            issues.append(f"{long_texts} texts longer than 1000 characters")
        
        # Check class imbalance
        if analysis['class_balance']['balance_ratio'] < 0.1:
            issues.append("Severe class imbalance detected")
        
        analysis['quality_issues'] = issues
        
        # Save analysis report
        self._save_quality_report(analysis)
        
        return analysis
    
    def _save_quality_report(self, analysis: Dict):
        """Save data quality analysis report"""
        report_path = self.artifact_dir / "reports" / f"data_quality_{self.artifact_version}.json"
        
        with open(report_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Quality report saved to {report_path}")
    
    def create_stratified_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create stratified train/validation/test splits with improved stratification
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing train, validation, and test DataFrames
        """
        logger.info("Creating stratified splits")
        
        # Create label encoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(df['label'])
        
        # Save label mapping
        label_mapping = {
            'label_to_id': {label: int(id_) for label, id_ in zip(label_encoder.classes_, range(len(label_encoder.classes_)))},
            'id_to_label': {int(id_): label for label, id_ in zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
        }
        
        with open(self.artifact_dir / "labels" / f"label_mapping_{self.artifact_version}.json", 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            df.index, y_encoded,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y_encoded
        )
        
        # Second split: separate validation from remaining data
        valid_size_relative = self.config['valid_size'] / (1 - self.config['test_size'])
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_temp, y_temp,
            test_size=valid_size_relative,
            random_state=self.config['random_state'],
            stratify=y_temp
        )
        
        # Create DataFrames for each split
        splits = {
            'train': df.loc[X_train].reset_index(drop=True),
            'valid': df.loc[X_valid].reset_index(drop=True),
            'test': df.loc[X_test].reset_index(drop=True)
        }
        
        # Save split indices
        for split_name, indices in [('train', X_train), ('valid', X_valid), ('test', X_test)]:
            pd.Series(indices).to_csv(
                self.artifact_dir / "splits" / f"{split_name}_idx_{self.artifact_version}.csv",
                index=False, header=False
            )
        
        # Log split information
        for split_name, split_df in splits.items():
            logger.info(f"{split_name}: {len(split_df)} samples")
            logger.info(f"{split_name} class distribution: {split_df['label'].value_counts().to_dict()}")
        
        self.splits = splits
        return splits
    
    def extract_enhanced_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Extract comprehensive feature set including traditional and advanced features
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing different feature types
        """
        logger.info("Extracting enhanced features")
        
        features = {}
        
        # 1. Sentiment features (VADER + additional metrics)
        if self.config.get('include_sentiment', True):
            features['sentiment'] = self._extract_sentiment_features(df['text'])
        
        # 2. POS features
        if self.config.get('include_pos', True):
            features['pos'] = self._extract_pos_features(df['text'])
        
        # 3. Readability features
        if self.config.get('include_readability', True):
            features['readability'] = self._extract_readability_features(df['text'])
        
        # 4. Lexical features
        if self.config.get('include_lexical', True):
            features['lexical'] = self._extract_lexical_features(df['text'])
        
        # 5. Statistical features
        if self.config.get('include_statistical', True):
            features['statistical'] = self._extract_statistical_features(df['text'])
        
        # 6. TF-IDF features
        features['tfidf'] = self._extract_tfidf_features(df)
        
        # 7. Sequence features for deep learning
        features['sequences'] = self._extract_sequence_features(df['text'])
        
        # 8. Transformer features
        features['transformers'] = self._extract_transformer_features(df['text'])
        
        self.features = features
        return features
    
    def _extract_sentiment_features(self, text_series: pd.Series) -> pd.DataFrame:
        """Extract comprehensive sentiment features"""
        logger.info("Extracting sentiment features")
        
        def get_sentiment_features(text):
            # VADER sentiment
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            
            # Additional sentiment metrics
            words = word_tokenize(text.lower())
            positive_words = sum(1 for word in words if word in self.sentiment_analyzer.lexicon and self.sentiment_analyzer.lexicon[word] > 0)
            negative_words = sum(1 for word in words if word in self.sentiment_analyzer.lexicon and self.sentiment_analyzer.lexicon[word] < 0)
            
            return pd.Series({
                'sent_neg': vader_scores['neg'],
                'sent_neu': vader_scores['neu'],
                'sent_pos': vader_scores['pos'],
                'sent_compound': vader_scores['compound'],
                'sent_positive_word_count': positive_words,
                'sent_negative_word_count': negative_words,
                'sent_positive_ratio': positive_words / max(len(words), 1),
                'sent_negative_ratio': negative_words / max(len(words), 1)
            })
        
        return text_series.apply(get_sentiment_features)
    
    def _extract_pos_features(self, text_series: pd.Series) -> pd.DataFrame:
        """Extract POS features using spaCy"""
        logger.info("Extracting POS features")
        
        if self.nlp is None:
            logger.warning("spaCy not available, skipping POS features")
            return pd.DataFrame(index=text_series.index)
        
        upos_tags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", 
                    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
        
        def get_pos_features(text):
            doc = self.nlp(text)
            pos_counts = Counter(token.pos_ for token in doc)
            total_tokens = len(doc)
            
            features = {}
            for tag in upos_tags:
                count = pos_counts.get(tag, 0)
                features[f'pos_{tag.lower()}_count'] = count
                features[f'pos_{tag.lower()}_ratio'] = count / max(total_tokens, 1)
            
            features['pos_total_tokens'] = total_tokens
            return pd.Series(features)
        
        return text_series.apply(get_pos_features)
    
    def _extract_readability_features(self, text_series: pd.Series) -> pd.DataFrame:
        """Extract readability features"""
        logger.info("Extracting readability features")
        
        def get_readability_features(text):
            try:
                return pd.Series({
                    'flesch_reading_ease': flesch_reading_ease(text),
                    'flesch_kincaid_grade': flesch_kincaid_grade(text),
                    'avg_sentence_length': len(text.split()) / max(len(sent_tokenize(text)), 1),
                    'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0
                })
            except:
                return pd.Series({
                    'flesch_reading_ease': 0,
                    'flesch_kincaid_grade': 0,
                    'avg_sentence_length': 0,
                    'avg_word_length': 0
                })
        
        return text_series.apply(get_readability_features)
    
    def _extract_lexical_features(self, text_series: pd.Series) -> pd.DataFrame:
        """Extract lexical features"""
        logger.info("Extracting lexical features")
        
        def get_lexical_features(text):
            words = word_tokenize(text.lower())
            sentences = sent_tokenize(text)
            
            # Basic counts
            word_count = len(words)
            sentence_count = len(sentences)
            char_count = len(text)
            
            # Vocabulary richness
            unique_words = len(set(words))
            lexical_diversity = unique_words / max(word_count, 1)
            
            # Stop word ratio
            stop_word_count = sum(1 for word in words if word in self.stop_words)
            stop_word_ratio = stop_word_count / max(word_count, 1)
            
            # Punctuation features
            punctuation_count = sum(1 for char in text if char in '.,!?;:')
            punctuation_ratio = punctuation_count / max(char_count, 1)
            
            # Capitalization features
            upper_count = sum(1 for char in text if char.isupper())
            upper_ratio = upper_count / max(char_count, 1)
            
            return pd.Series({
                'word_count': word_count,
                'sentence_count': sentence_count,
                'char_count': char_count,
                'unique_word_count': unique_words,
                'lexical_diversity': lexical_diversity,
                'stop_word_ratio': stop_word_ratio,
                'punctuation_count': punctuation_count,
                'punctuation_ratio': punctuation_ratio,
                'upper_char_count': upper_count,
                'upper_char_ratio': upper_ratio
            })
        
        return text_series.apply(get_lexical_features)
    
    def _extract_statistical_features(self, text_series: pd.Series) -> pd.DataFrame:
        """Extract statistical features"""
        logger.info("Extracting statistical features")
        
        def get_statistical_features(text):
            words = word_tokenize(text.lower())
            
            if not words:
                return pd.Series({
                    'word_length_mean': 0,
                    'word_length_std': 0,
                    'word_length_max': 0,
                    'word_length_min': 0,
                    'syllable_count': 0,
                    'syllable_ratio': 0
                })
            
            word_lengths = [len(word) for word in words]
            
            # Syllable counting (simplified)
            def count_syllables(word):
                vowels = 'aeiouy'
                word = word.lower()
                count = 0
                prev_was_vowel = False
                for char in word:
                    is_vowel = char in vowels
                    if is_vowel and not prev_was_vowel:
                        count += 1
                    prev_was_vowel = is_vowel
                return max(1, count)
            
            syllable_counts = [count_syllables(word) for word in words]
            
            return pd.Series({
                'word_length_mean': np.mean(word_lengths),
                'word_length_std': np.std(word_lengths),
                'word_length_max': max(word_lengths),
                'word_length_min': min(word_lengths),
                'syllable_count': sum(syllable_counts),
                'syllable_ratio': sum(syllable_counts) / len(words)
            })
        
        return text_series.apply(get_statistical_features)
    
    def _extract_tfidf_features(self, df: pd.DataFrame) -> Dict[str, sparse.csr_matrix]:
        """Extract TF-IDF features for each split"""
        logger.info("Extracting TF-IDF features")
        
        # Load splits
        train_idx = pd.read_csv(self.artifact_dir / "splits" / f"train_idx_{self.artifact_version}.csv", header=None).squeeze()
        valid_idx = pd.read_csv(self.artifact_dir / "splits" / f"valid_idx_{self.artifact_version}.csv", header=None).squeeze()
        test_idx = pd.read_csv(self.artifact_dir / "splits" / f"test_idx_{self.artifact_version}.csv", header=None).squeeze()
        
        # Extract texts for each split
        texts = {
            'train': df.loc[train_idx, 'text'].astype(str),
            'valid': df.loc[valid_idx, 'text'].astype(str),
            'test': df.loc[test_idx, 'text'].astype(str)
        }
        
        # Configure TF-IDF
        tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=self.config['tfidf_ngram_range'],
            min_df=self.config['tfidf_min_df'],
            max_df=self.config['tfidf_max_df'],
            max_features=self.config['tfidf_max_features'],
            sublinear_tf=self.config['tfidf_sublinear_tf'],
            norm=self.config['tfidf_norm']
        )
        
        # Fit on training data and transform all splits
        tfidf_matrices = {}
        tfidf_matrices['train'] = tfidf.fit_transform(texts['train'])
        tfidf_matrices['valid'] = tfidf.transform(texts['valid'])
        tfidf_matrices['test'] = tfidf.transform(texts['test'])
        
        # Save vectorizer and matrices
        joblib.dump(tfidf, self.artifact_dir / "features" / "tfidf" / f"tfidf_vectorizer_{self.artifact_version}.joblib")
        
        for split_name, matrix in tfidf_matrices.items():
            sparse.save_npz(
                self.artifact_dir / "features" / "tfidf" / f"X_{split_name}_tfidf_{self.artifact_version}.npz",
                matrix
            )
        
        return tfidf_matrices
    
    def _extract_sequence_features(self, text_series: pd.Series) -> Dict[str, np.ndarray]:
        """Extract sequence features for deep learning models"""
        logger.info("Extracting sequence features")
        
        # Load splits
        train_idx = pd.read_csv(self.artifact_dir / "splits" / f"train_idx_{self.artifact_version}.csv", header=None).squeeze()
        valid_idx = pd.read_csv(self.artifact_dir / "splits" / f"valid_idx_{self.artifact_version}.csv", header=None).squeeze()
        test_idx = pd.read_csv(self.artifact_dir / "splits" / f"test_idx_{self.artifact_version}.csv", header=None).squeeze()
        
        # Extract texts for each split
        texts = {
            'train': text_series.loc[train_idx].astype(str).tolist(),
            'valid': text_series.loc[valid_idx].astype(str).tolist(),
            'test': text_series.loc[test_idx].astype(str).tolist()
        }
        
        # Initialize tokenizer
        tokenizer = Tokenizer(
            num_words=self.config['vocab_size'],
            oov_token='<UNK>',
            lower=True
        )
        
        # Fit on training data
        tokenizer.fit_on_texts(texts['train'])
        
        # Convert to sequences
        sequences = {}
        for split_name, split_texts in texts.items():
            sequences[split_name] = tokenizer.texts_to_sequences(split_texts)
        
        # Determine max length
        if self.config.get('max_len') is None:
            all_lengths = [len(seq) for seq in sequences['train']]
            max_len = int(np.percentile(all_lengths, self.config.get('max_len_percentile', 95)))
            max_len = max(8, min(max_len, 256))
        else:
            max_len = self.config['max_len']
        
        # Pad sequences
        padded_sequences = {}
        for split_name, split_sequences in sequences.items():
            padded_sequences[split_name] = pad_sequences(
                split_sequences,
                maxlen=max_len,
                padding='post',
                truncating='post'
            )
        
        # Save tokenizer and sequences
        joblib.dump(tokenizer, self.artifact_dir / "features" / "sequences" / f"tokenizer_{self.artifact_version}.joblib")
        
        for split_name, sequences in padded_sequences.items():
            np.savez_compressed(
                self.artifact_dir / "features" / "sequences" / f"X_{split_name}_seq_{self.artifact_version}.npz",
                sequences
            )
        
        # Save metadata
        with open(self.artifact_dir / "features" / "sequences" / f"sequence_meta_{self.artifact_version}.txt", 'w') as f:
            f.write(f"VOCAB_SIZE={self.config['vocab_size']}\n")
            f.write(f"MAX_LEN={max_len}\n")
        
        return padded_sequences
    
    def _extract_transformer_features(self, text_series: pd.Series) -> Dict[str, np.ndarray]:
        """Extract transformer features"""
        logger.info("Extracting transformer features")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config['hf_model_name'])
        
        # Load splits
        train_idx = pd.read_csv(self.artifact_dir / "splits" / f"train_idx_{self.artifact_version}.csv", header=None).squeeze()
        valid_idx = pd.read_csv(self.artifact_dir / "splits" / f"valid_idx_{self.artifact_version}.csv", header=None).squeeze()
        test_idx = pd.read_csv(self.artifact_dir / "splits" / f"test_idx_{self.artifact_version}.csv", header=None).squeeze()
        
        # Extract texts for each split
        texts = {
            'train': text_series.loc[train_idx].astype(str).tolist(),
            'valid': text_series.loc[valid_idx].astype(str).tolist(),
            'test': text_series.loc[test_idx].astype(str).tolist()
        }
        
        def tokenize_batch(texts):
            encoding = tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.config['transformer_max_length'],
                return_attention_mask=True,
                return_tensors=None
            )
            return (
                np.array(encoding['input_ids'], dtype='int32'),
                np.array(encoding['attention_mask'], dtype='int32')
            )
        
        # Tokenize each split
        transformer_features = {}
        for split_name, split_texts in texts.items():
            input_ids, attention_mask = tokenize_batch(split_texts)
            transformer_features[f'{split_name}_input_ids'] = input_ids
            transformer_features[f'{split_name}_attention_mask'] = attention_mask
        
        # Save features
        for feature_name, feature_array in transformer_features.items():
            np.savez_compressed(
                self.artifact_dir / "features" / "transformers" / f"{feature_name}_{self.artifact_version}.npz",
                feature_array
            )
        
        # Save model name
        with open(self.artifact_dir / "features" / "transformers" / f"hf_model_name_{self.artifact_version}.txt", 'w') as f:
            f.write(self.config['hf_model_name'])
        
        return transformer_features
    
    def apply_data_augmentation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data augmentation to balance classes
        
        Args:
            df: Input DataFrame
            
        Returns:
            Augmented DataFrame
        """
        if not self.config.get('enable_augmentation', False) or self.augmenter is None:
            logger.info("Data augmentation disabled")
            return df
        
        logger.info("Applying data augmentation")
        
        # Calculate target sample count (use median class size)
        class_counts = df['label'].value_counts()
        target_count = int(class_counts.median())
        
        augmented_rows = []
        
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            current_count = len(label_df)
            
            if current_count < target_count:
                # Calculate how many samples to generate
                samples_needed = target_count - current_count
                
                # Sample with replacement for augmentation
                samples_to_augment = label_df.sample(n=min(samples_needed, len(label_df)), replace=True)
                
                for _, row in samples_to_augment.iterrows():
                    try:
                        # Apply augmentation
                        augmented_text = self.augmenter.augment(row['text'])
                        
                        augmented_rows.append({
                            'text': augmented_text,
                            'label': row['label'],
                            'source': f"{row.get('source', 'unknown')}_augmented"
                        })
                    except Exception as e:
                        logger.warning(f"Augmentation failed for text: {e}")
                        continue
        
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            combined_df = pd.concat([df, augmented_df], ignore_index=True)
            logger.info(f"Augmentation complete: {len(df)} -> {len(combined_df)} samples")
            return combined_df
        else:
            logger.info("No augmentation applied")
            return df
    
    def save_processed_data(self, df: pd.DataFrame, features: Dict):
        """
        Save all processed data and features
        
        Args:
            df: Processed DataFrame
            features: Dictionary of extracted features
        """
        logger.info("Saving processed data and features")
        
        # Save main dataset
        df.to_csv(
            self.artifact_dir / f"processed_data_{self.artifact_version}.tsv",
            sep='\t', index=False
        )
        
        # Save dense features
        dense_features = []
        for feature_type, feature_df in features.items():
            if isinstance(feature_df, pd.DataFrame) and not feature_df.empty:
                dense_features.append(feature_df)
        
        if dense_features:
            combined_dense = pd.concat(dense_features, axis=1)
            combined_dense.to_parquet(
                self.artifact_dir / "features" / "dense" / f"dense_features_{self.artifact_version}.parquet"
            )
            
            # Save feature names
            with open(self.artifact_dir / "features" / "dense" / f"dense_feature_columns_{self.artifact_version}.json", 'w') as f:
                json.dump(list(combined_dense.columns), f, indent=2)
        
        # Fit and save scaler for dense features
        if dense_features:
            train_idx = pd.read_csv(self.artifact_dir / "splits" / f"train_idx_{self.artifact_version}.csv", header=None).squeeze()
            scaler = MaxAbsScaler()
            scaler.fit(combined_dense.loc[train_idx])
            joblib.dump(scaler, self.artifact_dir / "features" / "dense" / f"dense_scaler_{self.artifact_version}.joblib")
        
        # Save labels for each split
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(df['label'])
        
        for split_name in ['train', 'valid', 'test']:
            split_idx = pd.read_csv(self.artifact_dir / "splits" / f"{split_name}_idx_{self.artifact_version}.csv", header=None).squeeze()
            split_labels = y_encoded[split_idx]
            pd.Series(split_labels).to_csv(
                self.artifact_dir / "labels" / f"y_{split_name}_{self.artifact_version}.csv",
                index=False, header=False
            )
        
        logger.info("Data preparation complete!")


def main():
    """Main function to run the improved data preparation pipeline"""
    
    # Configuration
    config = {
        # Paths
        'artifact_dir': 'artifacts',
        'artifact_version': 'v2',
        
        # Data splitting
        'test_size': 0.10,
        'valid_size': 0.10,
        'random_state': 42,
        
        # Feature extraction
        'include_sentiment': True,
        'include_pos': True,
        'include_readability': True,
        'include_lexical': True,
        'include_statistical': True,
        
        # TF-IDF parameters
        'tfidf_ngram_range': (1, 2),
        'tfidf_min_df': 5,
        'tfidf_max_df': 0.8,
        'tfidf_max_features': 200000,
        'tfidf_sublinear_tf': True,
        'tfidf_norm': 'l2',
        
        # Sequence parameters
        'vocab_size': 40000,
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
    
    # Define dataset paths (update these with your actual file paths)
    dataset_paths = {
        'dailydialog': 'daily_dialog.csv',
        'isear': 'ISEAR.csv',
        'goemotions_1': 'goemotions_1.csv',
        'goemotions_2': 'goemotions_2.csv',
        'goemotions_3': 'goemotions_3.csv'
    }
    
    try:
        # Load and validate data
        df = pipeline.load_and_validate_data(dataset_paths)
        
        # Analyze data quality
        quality_analysis = pipeline.analyze_data_quality(df)
        print("\nData Quality Analysis:")
        print(f"Total samples: {quality_analysis['total_samples']}")
        print(f"Class balance ratio: {quality_analysis['class_balance']['balance_ratio']:.3f}")
        print(f"Quality issues: {quality_analysis['quality_issues']}")
        
        # Apply data augmentation if needed
        if quality_analysis['class_balance']['balance_ratio'] < 0.3:
            df = pipeline.apply_data_augmentation(df)
        
        # Create stratified splits
        splits = pipeline.create_stratified_splits(df)
        
        # Extract features
        features = pipeline.extract_enhanced_features(df)
        
        # Save everything
        pipeline.save_processed_data(df, features)
        
        print("\nData preparation completed successfully!")
        print(f"Artifacts saved to: {config['artifact_dir']}")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()