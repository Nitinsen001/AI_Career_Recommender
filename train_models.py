#!/usr/bin/env python3
"""
AI Career Recommender - ENHANCED MODEL TRAINING SCRIPT
================================================
LIBRARY USAGES:
- pandas, numpy: Data manipulation and numerical operations
- joblib: Model saving/loading
- warnings: Handling warnings
- os: File system operations
- re: Regular expressions for text cleaning
- json: JSON serialization
- datetime: Timestamp handling
- sklearn: Machine learning models and utilities
- nltk: Natural Language Processing
- imblearn: Handling imbalanced datasets
- xgboost: Gradient boosting algorithm
"""

import os
import re
import json
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

# ==============================================================================
# MACHINE LEARNING LIBRARIES FROM scikit-learn
# ==============================================================================
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# ==============================================================================
# IMBALANCED LEARNING LIBRARIES
# ==============================================================================
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ==============================================================================
# GRADIENT BOOSTING LIBRARY
# ==============================================================================
from xgboost import XGBClassifier

# ==============================================================================
# NATURAL LANGUAGE PROCESSING LIBRARIES
# ==============================================================================
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

warnings.filterwarnings('ignore')

# --- Dependency diagnostics to help troubleshoot environment issues ---
try:
    from importlib.metadata import version, PackageNotFoundError
except Exception:
    try:
        from pkg_resources import get_distribution as _get_dist

        def version(pkg):
            try:
                return _get_dist(pkg).version
            except Exception:
                raise PackageNotFoundError
    except Exception:
        def version(pkg):
            raise Exception("Cannot determine package versions on this Python environment.")

    class PackageNotFoundError(Exception):
        pass


def dependency_diagnostics():
    """Print versions of key libraries to help debug imbalanced-learn warnings."""
    print("\n[DIAG] Checking installed package versions:")
    pkgs = ['imbalanced-learn', 'scikit-learn', 'numpy', 'xgboost', 'pandas']
    for p in pkgs:
        try:
            v = version(p)
        except PackageNotFoundError:
            v = 'NOT INSTALLED'
        except Exception:
            v = 'UNKNOWN'
        print(f"[DIAG] {p}: {v}")
    print("[DIAG] If imbalanced-learn is installed but you still see warnings, you may have multiple Python environments or an incompatible scikit-learn version.")
    print("[DIAG] Quick fixes:")
    print("  • Ensure you run the same Python interpreter where you installed the package.")
    print("  • Upgrade/install compatible versions: pip install -U scikit-learn imbalanced-learn")
    print("  • If using conda: conda install -c conda-forge imbalanced-learn scikit-learn")
    print("")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models')
ENHANCED_MODEL_DIR = os.path.join(BASE_DIR, 'enhanced_models')

# Create directories if they don't exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ENHANCED_MODEL_DIR, exist_ok=True)

# ==============================================================================
# ENHANCED TEXT PREPROCESSING (ALL MODELS)
# ==============================================================================
class AdvancedTextPreprocessor:
    """Advanced text preprocessing for all text-based models"""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Comprehensive text cleaning"""
        if pd.isna(text) or not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'http\S+|www\S+|[0-9]+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        return ' '.join(tokens)

# ==============================================================================
# DATA LOADING FUNCTION
# ==============================================================================

def load_datasets():
    """Load all required datasets (6 files)"""
    print("\n" + "=" * 80)
    print("AI Career Recommender - ENHANCED MODEL TRAINING STARTED")
    print("=" * 80)
    print("[DATA] Loading datasets...")

    try:
        career_df = pd.read_csv(os.path.join(DATASET_DIR, 'career_dataset.csv'))
        skills_df = pd.read_csv(os.path.join(DATASET_DIR, 'skills_dataset.csv'))
        market_df = pd.read_csv(os.path.join(DATASET_DIR, 'job_market.csv'))
        personality_ocean_df = pd.read_csv(os.path.join(DATASET_DIR, 'personality.csv'))
        personality_mbti_df = pd.read_csv(os.path.join(DATASET_DIR, 'mbti_training_data.csv'))
        resume_df = pd.read_csv(os.path.join(DATASET_DIR, 'Resume.csv'))

        print(f"[OK] Career dataset: {career_df.shape}")
        print(f"[OK] Skills dataset: {skills_df.shape}")
        print(f"[OK] Personality (OCEAN) dataset: {personality_ocean_df.shape}")
        print(f"[OK] MBTI Training dataset: {personality_mbti_df.shape}")
        print(f"[OK] Resume dataset: {resume_df.shape}")

        return career_df, skills_df, personality_ocean_df, market_df, personality_mbti_df, resume_df

    except Exception as e:
        print(f"[ERROR] Error loading datasets: {e}")
        print("Please ensure all 6 files are in the 'datasets/' folder.")
        return None, None, None, None, None, None

# ==============================================================================
# MODEL 1: ENHANCED RESUME CLASSIFIER
# ==============================================================================

# def train_enhanced_resume_classifier(resume_df):
#     """ENHANCED Resume Classification Model with Ensemble and SMOTE"""
#     print("\n" + "=" * 60)
#     print("MODEL 1: ENHANCED RESUME CLASSIFIER TRAINING")
#     print("=" * 60)

#     preprocessor = AdvancedTextPreprocessor()

#     print("[RESUME] Applying advanced text preprocessing...")
#     resume_df['cleaned_text'] = resume_df['resume_text'].apply(preprocessor.clean_text)

#     X = resume_df['cleaned_text']
#     y = resume_df['category']

#     min_samples = 5
#     valid_categories = y.value_counts()[y.value_counts() >= min_samples].index
#     resume_df_filtered = resume_df[y.isin(valid_categories)]

#     X_filtered = resume_df_filtered['cleaned_text']
#     y_filtered = resume_df_filtered['category']

#     print(f"[RESUME] After filtering: {X_filtered.shape[0]} samples, {y_filtered.nunique()} categories")

#     if len(X_filtered) < 100:
#         print("[WARNING] Not enough filtered resume data to train. Skipping.")
#         return None

#     le = LabelEncoder()
#     y_encoded = le.fit_transform(y_filtered)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_filtered, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#     )

#     print("[RESUME] Building enhanced pipeline with ensemble and SMOTE...")

#     smote = SMOTE(k_neighbors=1, random_state=42)

#     resume_pipeline = ImbPipeline([
#         ('tfidf', TfidfVectorizer(
#             max_features=5000,
#             ngram_range=(1, 2),
#             stop_words='english',
#             min_df=2,
#             max_df=0.9,
#             sublinear_tf=True,
#         )),
#         ('smote', smote),
#         ('classifier', XGBClassifier(
#             random_state=42,
#             eval_metric='logloss',
#             tree_method='hist',
#             max_depth=5,
#             learning_rate=0.15,
#             n_estimators=100,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             n_jobs=2,
#         )),
#     ])

#     print("[RESUME] Performing cross-validation...")
#     skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#     cv_scores = cross_val_score(resume_pipeline, X_train, y_train, cv=skf, scoring='accuracy', error_score='raise')
#     print(f"[RESUME] Cross-validation scores: {cv_scores}")
#     print(f"[RESUME] Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

#     y_train_series = pd.Series(y_train)
#     vc = y_train_series.value_counts()
#     keep_labels = vc[vc >= 3].index
#     mask = y_train_series.isin(keep_labels)
#     if mask.sum() != len(y_train_series):
#         print(f"[RESUME] Filtering ultra-rare classes in training: {len(y_train_series) - mask.sum()} samples removed")
#         X_train = X_train[mask]
#         y_train = y_train_series[mask]

#     print("[RESUME] Training final ensemble model...")
#     resume_pipeline.fit(X_train, y_train)

#     print("[RESUME] Comprehensive model evaluation...")
#     y_pred = resume_pipeline.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

#     print(f"[RESUME] Final Test Accuracy: {accuracy:.3f}")
#     print(f"[RESUME] Precision: {precision:.3f}")
#     print(f"[RESUME] Recall: {recall:.3f}")
#     print(f"[RESUME] F1-Score: {f1:.3f}")

#     model_metadata = {
#         'model': resume_pipeline,
#         'preprocessor': preprocessor,
#         'label_encoder': le,
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1_score': f1,
#         'cv_scores': cv_scores.tolist(),
#         'training_date': datetime.now().isoformat(),
#         'feature_count': 8000,
#         'classes': le.classes_.tolist(),
#     }

#     model_path = os.path.join(ENHANCED_MODEL_DIR, 'enhanced_resume_classifier.pkl')
#     joblib.dump(model_metadata, model_path)
#     print(f"[SAVE] Enhanced Resume Classifier saved to: {model_path}")

#     return model_metadata

# # ==============================================================================
# # MODEL 2: ENHANCED PERSONALITY CLASSIFIER (MBTI)
# # ==============================================================================

# def train_enhanced_personality_classifier(personality_mbti_df):
#     """ENHANCED MBTI Personality Classifier with Advanced Features"""
#     print("\n" + "=" * 60)
#     print("MODEL 2: ENHANCED PERSONALITY CLASSIFIER (MBTI)")
#     print("=" * 60)

#     preprocessor = AdvancedTextPreprocessor()

#     print("[PERSONALITY] Applying advanced text preprocessing...")
#     personality_mbti_df['cleaned_posts'] = personality_mbti_df['posts'].apply(preprocessor.clean_text)

#     X = personality_mbti_df['cleaned_posts']
#     y = personality_mbti_df['type']

#     le = LabelEncoder()
#     y_encoded = le.fit_transform(y)

#     print(f"[PERSONALITY] Unique personality types: {len(le.classes_)}")
#     print(f"[PERSONALITY] Classes: {le.classes_}")

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
#     )

#     print("[PERSONALITY] Building enhanced ensemble pipeline...")

#     ensemble_classifier = VotingClassifier(
#         estimators=[
#             ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
#             ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
#             ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
#         ],
#         voting='soft',
#     )

#     personality_pipeline = ImbPipeline([
#         ('tfidf', TfidfVectorizer(
#             max_features=10000,
#             ngram_range=(1, 3),
#             stop_words='english',
#             min_df=3,
#             max_df=0.85,
#             sublinear_tf=True,
#         )),
#         ('smote', SMOTE(random_state=42)),
#         ('classifier', ensemble_classifier),
#     ])

#     print("[PERSONALITY] Performing cross-validation...")
#     cv_scores = cross_val_score(personality_pipeline, X_train, y_train, cv=5, scoring='accuracy')
#     print(f"[PERSONALITY] Cross-validation scores: {cv_scores}")
#     print(f"[PERSONALITY] Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

#     print("[PERSONALITY] Training final ensemble model...")
#     personality_pipeline.fit(X_train, y_train)

#     print("[PERSONALITY] Comprehensive model evaluation...")
#     y_pred = personality_pipeline.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

#     print(f"[PERSONALITY] Final Test Accuracy: {accuracy:.3f}")
#     print(f"[PERSONALITY] Precision: {precision:.3f}")
#     print(f"[PERSONALITY] Recall: {recall:.3f}")
#     print(f"[PERSONALITY] F1-Score: {f1:.3f}")

#     model_metadata = {
#         'model': personality_pipeline,
#         'preprocessor': preprocessor,
#         'label_encoder': le,
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1_score': f1,
#         'cv_scores': cv_scores.tolist(),
#         'training_date': datetime.now().isoformat(),
#         'feature_count': 10000,
#         'classes': le.classes_.tolist(),
#     }

#     model_path = os.path.join(ENHANCED_MODEL_DIR, 'enhanced_personality_classifier.pkl')
#     joblib.dump(model_metadata, model_path)
#     print(f"[SAVE] Enhanced Personality Classifier saved to: {model_path}")

#     return model_metadata

# ==============================================================================
# MODEL 3: ENHANCED TEXT VECTORIZER
# ==============================================================================

def train_enhanced_text_vectorizer(career_df, skills_df, personality_mbti_df, resume_df):
    """ENHANCED Global Text Vectorizer with Advanced Features"""
    print("\n" + "=" * 60)
    print("MODEL 3: ENHANCED TEXT VECTORIZER")
    print("=" * 60)

    preprocessor = AdvancedTextPreprocessor()

    text_data = []

    if 'description' in career_df.columns:
        career_texts = career_df['description'].astype(str).apply(preprocessor.clean_text)
        text_data.extend(career_texts.tolist())

    if 'skill_name' in skills_df.columns:
        skill_texts = skills_df['skill_name'].astype(str).apply(preprocessor.clean_text)
        text_data.extend(skill_texts.tolist())

    if 'posts' in personality_mbti_df.columns:
        personality_texts = personality_mbti_df['posts'].astype(str).apply(preprocessor.clean_text)
        text_data.extend(personality_texts.tolist())

    if 'resume_text' in resume_df.columns:
        resume_texts = resume_df['resume_text'].astype(str).apply(preprocessor.clean_text)
        text_data.extend(resume_texts.tolist())

    text_data = [t for t in text_data if len(t.strip()) > 10]

    print(f"[VECTORIZER] Total text samples: {len(text_data)}")
    print(f"[VECTORIZER] Sample texts: {text_data[:2]}")

    print("[VECTORIZER] Training enhanced TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 3),
        stop_words='english',
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
    )

    vectorizer.fit(text_data)

    print(f"[VECTORIZER] Vocabulary size: {len(vectorizer.vocabulary_)}")

    print("[VECTORIZER] Applying dimensionality reduction...")

    text_vectors = vectorizer.transform(text_data)
    svd = TruncatedSVD(n_components=500, random_state=42)
    reduced_vectors = svd.fit_transform(text_vectors)

    print(f"[VECTORIZER] Original dimensions: {text_vectors.shape[1]}")
    print(f"[VECTORIZER] Reduced dimensions: {reduced_vectors.shape[1]}")
    print(f"[VECTORIZER] Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")

    vectorizer_metadata = {
        'vectorizer': vectorizer,
        'svd': svd,
        'preprocessor': preprocessor,
        'vocabulary_size': len(vectorizer.vocabulary_),
        'feature_names': vectorizer.get_feature_names_out().tolist(),
        'explained_variance': svd.explained_variance_ratio_.sum(),
        'training_date': datetime.now().isoformat(),
    }

    vectorizer_path = os.path.join(ENHANCED_MODEL_DIR, 'enhanced_vectorizer.pkl')
    joblib.dump(vectorizer_metadata, vectorizer_path)
    print(f"[SAVE] Enhanced Text Vectorizer saved to: {vectorizer_path}")

    return vectorizer_metadata

# ==============================================================================
# MODEL 4: ENHANCED SKILL EXTRACTOR
# ==============================================================================

def train_enhanced_skill_extractor(skills_df):
    """ENHANCED Rule-Based Skill Extractor with Advanced Matching"""
    print("\n" + "=" * 60)
    print("MODEL 4: ENHANCED SKILL EXTRACTOR")
    print("=" * 60)

    if 'skill_name' not in skills_df.columns or 'category' not in skills_df.columns:
        print("[ERROR] Skills DF missing essential columns. Skipping Skill Extractor.")
        return None

    skills_mapping = {}
    category_mapping = {}
    proficiency_mapping = {}

    print("[SKILLS] Building enhanced skills database...")

    for _, row in skills_df.iterrows():
        skill_name = str(row['skill_name']).strip().lower()
        category = str(row['category'])

        skills_mapping[skill_name] = category

        words = skill_name.split()
        if len(words) > 1:
            acronym = ''.join([word[0] for word in words if len(word) > 0])
            if acronym not in skills_mapping:
                skills_mapping[acronym] = category

            if skill_name.endswith('s') and skill_name[:-1] not in skills_mapping:
                skills_mapping[skill_name[:-1]] = category

        category_mapping[skill_name] = category
        if 'proficiency_levels' in skills_df.columns:
            proficiency_mapping[skill_name] = str(row['proficiency_levels'])

    print(f"[SKILLS] Total skills in database: {len(skills_mapping)}")
    print(f"[SKILLS] Sample skills: {list(skills_mapping.keys())[:5]}")

    extractor_data = {
        'skills_mapping': skills_mapping,
        'category_mapping': category_mapping,
        'proficiency_mapping': proficiency_mapping,
        'total_skills': len(skills_mapping),
        'created_date': datetime.now().isoformat(),
    }

    extractor_path = os.path.join(ENHANCED_MODEL_DIR, 'enhanced_skill_extractor.pkl')
    joblib.dump(extractor_data, extractor_path)
    print(f"[SAVE] Enhanced Skill Extractor saved to: {extractor_path}")

    return extractor_data

# ==============================================================================
# MODEL 5: ENHANCED CONFIDENCE SCORER (CAREER RECOMMENDER CORE)
# ==============================================================================

def train_enhanced_confidence_scorer(career_df, vectorizer_metadata):
    """ENHANCED Confidence Scoring System for Career Recommendations"""
    print("\n" + "=" * 60)
    print("MODEL 5: ENHANCED CONFIDENCE SCORER")
    print("=" * 60)

    vectorizer = vectorizer_metadata['vectorizer']
    svd = vectorizer_metadata['svd']
    preprocessor = vectorizer_metadata['preprocessor']

    print("[SCORER] Creating enhanced career profiles...")

    career_profiles = []
    career_details = []

    for idx, career in career_df.iterrows():
        profile_text = ""
        if 'career_name' in career_df.columns:
            profile_text += f" {career['career_name']}"
        if 'description' in career_df.columns:
            profile_text += f" {career['description']}"
        if 'required_skills' in career_df.columns:
            profile_text += f" {career['required_skills']}"
        if 'domain' in career_df.columns:
            profile_text += f" {career['domain']}"

        cleaned_text = preprocessor.clean_text(profile_text)
        career_profiles.append(cleaned_text)

        career_details.append({
            'career_id': career['career_id'] if 'career_id' in career_df.columns else idx,
            'career_name': career['career_name'] if 'career_name' in career_df.columns else "Unknown",
            'domain': career['domain'] if 'domain' in career_df.columns else "Unknown",
            'average_salary': career['average_salary'] if 'average_salary' in career_df.columns else 0,
            'job_growth_rate': career['job_growth_rate'] if 'job_growth_rate' in career_df.columns else 0,
        })

    print(f"[SCORER] Processed {len(career_profiles)} career profiles")

    print("[SCORER] Transforming career profiles to vector space...")

    career_vectors_tfidf = vectorizer.transform(career_profiles)
    career_vectors_reduced = svd.transform(career_vectors_tfidf)

    print(f"[SCORER] Career vectors shape: {career_vectors_reduced.shape}")

    print("[SCORER] Calculating career similarity matrix...")
    career_similarity = cosine_similarity(career_vectors_reduced)

    scorer_data = {
        'vectorizer': vectorizer,
        'svd': svd,
        'preprocessor': preprocessor,
        'career_vectors': career_vectors_reduced,
        'career_similarity': career_similarity,
        'career_details': career_details,
        'career_profiles': career_profiles,
        'training_date': datetime.now().isoformat(),
        'total_careers': len(career_profiles),
    }

    scorer_path = os.path.join(ENHANCED_MODEL_DIR, 'enhanced_confidence_scorer.pkl')
    joblib.dump(scorer_data, scorer_path)
    print(f"[SAVE] Enhanced Confidence Scorer saved to: {scorer_path}")

    return scorer_data

# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def main():
    """Main training function to orchestrate all enhanced model training"""
    print("\n" + "=" * 80)
    print("AI CAREER RECOMMENDER - ENHANCED MODEL TRAINING")
    print("=" * 80)

    try:
        dependency_diagnostics()
    except Exception as e:
        print(f"[DIAG] Could not run dependency diagnostics: {e}")

    datasets = load_datasets()
    if any(df is None for df in datasets):
        print("\n[FATAL ERROR] Data loading failed. Training stopped.")
        return

    career_df, skills_df, personality_ocean_df, market_df, personality_mbti_df, resume_df = datasets

    trained_models = {}

    try:
        # resume_model = train_enhanced_resume_classifier(resume_df)
        # trained_models['resume_classifier'] = resume_model

        # personality_model = train_enhanced_personality_classifier(personality_mbti_df)
        # trained_models['personality_classifier'] = personality_model

        vectorizer_model = train_enhanced_text_vectorizer(career_df, skills_df, personality_mbti_df, resume_df)
        trained_models['text_vectorizer'] = vectorizer_model

        skill_extractor = train_enhanced_skill_extractor(skills_df)
        trained_models['skill_extractor'] = skill_extractor

        if vectorizer_model is not None:
            confidence_scorer = train_enhanced_confidence_scorer(career_df, vectorizer_model)
            trained_models['confidence_scorer'] = confidence_scorer

        print("\n" + "=" * 80)
        print("TRAINING SUMMARY - ENHANCED MODELS")
        print("=" * 80)

        for model_name, model_data in trained_models.items():
            if model_data and isinstance(model_data, dict) and 'accuracy' in model_data:
                print(f"\u2713 {model_name.upper()}: Accuracy = {model_data['accuracy']:.3f}")
            elif model_data:
                print(f"\u2713 {model_name.upper()}: Trained successfully")
            else:
                print(f"\u2717 {model_name.upper()}: Skipped or failed")

        print(f"\n[SUCCESS] All {len(trained_models)} enhanced models trained and saved!")
        print(f"Models saved in: {ENHANCED_MODEL_DIR}/")

    except Exception as e:
        print(f"\n[FATAL ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
