import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from datetime import datetime
import os
import gc
import time
import psutil
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from transformers import RobertaTokenizer, RobertaModel
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import XLNetTokenizer, XLNetModel
from transformers import DebertaTokenizer, DebertaModel
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
import itertools

class BaselineExperiment:
    def __init__(self, data_dir, output_dir):
        """
        Initialize experiment setup
        
        Args:
            data_dir: Directory containing the group_{i}_train.csv and group_{i}_test.csv files
            output_dir: Base directory for results
        """
        self.data_dir = data_dir
        
        # Find next available test folder number
        test_num = 1
        while os.path.exists(os.path.join(output_dir, f'test_{test_num}')):
            test_num += 1
        
        # Create new test folder
        self.output_dir = os.path.join(output_dir, f'test_{test_num}')
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Results will be saved to: {self.output_dir}")
        
        # Set up GPU devices - USING MORE GPU RESOURCES
        self.use_cuda = torch.cuda.is_available()
        self.multi_gpu = torch.cuda.device_count() > 1  # Enable multi-GPU if available
        print(f"CUDA available: {self.use_cuda}, Using multi-GPU: {self.multi_gpu}")
        
        if self.use_cuda:
            # Use all available GPUs
            self.device = torch.device('cuda:0')  # Use primary GPU
            self.gpu_devices = list(range(torch.cuda.device_count()))
            print(f"Using GPU(s): {[torch.cuda.get_device_name(i) for i in self.gpu_devices]}")
            
            # Get GPU memory info
            for i in self.gpu_devices:
                print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            
            # Clear cache before starting
            torch.cuda.empty_cache()
            
            # Set higher memory limit
            torch.cuda.set_per_process_memory_fraction(0.9, 0)  # Use 90% of GPU memory
            print("Memory usage increased to 90% to utilize more GPU resources")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
        
        # Increased batch size
        self.batch_size = 32  # Increased from 8 to 32
        print(f"Using larger batch size: {self.batch_size}")
        
        # Increased max sequence length
        self.max_seq_length = 256  # Increased from 128 to 256
        print(f"Using increased max sequence length: {self.max_seq_length}")
        
        # Target variables to predict with correct column names
        self.targets = [
            'Individual_level_PA_State', 
            # 'Individual_level_happy_State', 
            # 'Individual_level_cheerful_State', 
            # 'Individual_level_pleased_State',
            # 'Individual_level_grateful_State', 
            # 'Individual_level_interactions_quality_State', 
            'Individual_level_NA_State',
            # 'Individual_level_sad_State', 
            # 'Individual_level_afraid_State', 
            # 'Individual_level_miserable_State', 
            # 'Individual_level_worried_State',
            # 'Individual_level_lonely_State', 
            # 'Individual_level_pain_State', 
            'Individual_level_ER_desire_State',
            'INT_availability'
        ]
        
        # Initialize results dictionary with multitask structure
        self.results = {}
        
        # Create checkpoint directory
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Set target types based on the data inspector results
        # From analysis, all targets are classification
        self.target_types = {}
        for target in self.targets:
            self.target_types[target] = 'classification'
        
        print(f"Target types set: {self.target_types}")
    
    def check_system_resources(self):
        """Check system resources to prevent crashes"""
        # Check CPU usage
        cpu_percent = psutil.cpu_percent()
        # Check memory usage
        memory_percent = psutil.virtual_memory().percent
        # Check GPU memory if available
        if self.use_cuda:
            gpu_memory_used = torch.cuda.memory_allocated(self.device) / torch.cuda.get_device_properties(self.device.index).total_memory
            gpu_memory_percent = gpu_memory_used * 100
            
            print(f"System resources: CPU {cpu_percent}%, RAM {memory_percent}%, GPU {gpu_memory_percent:.1f}%")
            
            # If resources are high, force garbage collection and wait
            if cpu_percent > 90 or memory_percent > 90 or gpu_memory_percent > 80:
                print("High resource usage detected! Cleaning memory and cooling down...")
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(10)
                return False
        else:
            print(f"System resources: CPU {cpu_percent}%, RAM {memory_percent}%")
            if cpu_percent > 90 or memory_percent > 90:
                print("High resource usage detected! Cleaning memory and cooling down...")
                gc.collect()
                time.sleep(10)
                return False
        
        return True
    
    def save_checkpoint(self, model_name, target=None):
        """Save checkpoint to resume from crashes"""
        checkpoint = {
            'model_name': model_name,
            'target': target,
            'results': self.results,
            'completed_models': getattr(self, 'completed_models', [])
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_group_data(self, group_num):
        """Load and preprocess train and test data for a specific group"""
        train_path = os.path.join(self.data_dir, f'group_{group_num}_train.csv')
        test_path = os.path.join(self.data_dir, f'group_{group_num}_test.csv')
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Basic preprocessing
        for df in [train_df, test_df]:
            df['emotion_driver'] = df['emotion_driver'].fillna('')
            df['emotion_driver'] = df['emotion_driver'].str.strip()
            
            # Convert all target variables to proper format
            for target in self.targets:
                if target in df.columns:
                    # For boolean columns, convert to integers (0/1)
                    if df[target].dtype == 'bool':
                        df[target] = df[target].astype(int)
                    
                    # For INT_availability (yes/no format)
                    elif target == 'INT_availability':
                        df[target] = (df[target].str.lower() == 'yes').astype(int)
                    
                    # For any other format, try to convert to proper binary values
                    else:
                        # Try to convert to numeric if string
                        if df[target].dtype == 'object':
                            # For yes/no, true/false strings
                            df[target] = df[target].str.lower().map(
                                lambda x: 1 if x in ['yes', 'true', '1', 'y', 't'] else 0 if pd.notna(x) else np.nan
                            )
                        
                        # Ensure all values are 0 or 1
                        if df[target].dtype != 'int64':
                            df[target] = pd.to_numeric(df[target], errors='coerce')
                            df[target] = (df[target] > 0.5).astype(int)
        
        return train_df, test_df
    
    def evaluate_traditional_ml(self, model_name, vectorizer, classifier):
        """
        Evaluate traditional ML models (BoW, TF-IDF) using multitask approach
        """
        print(f"Evaluating {model_name} with multitask approach...")
        fold_scores = {target: {'acc': [], 'auc': []} for target in self.targets}
        combined_scores = {'acc': [], 'auc': []}
        
        # Create a multi-output classifier (all targets are classification)
        multi_classifier = MultiOutputClassifier(classifier, n_jobs=-1)
        
        for group in range(1, 6):  # 5 groups
            # Load data for this group
            train_df, test_df = self.load_group_data(group)
            
            # Prepare data
            X_train = train_df['emotion_driver'].values
            X_test = test_df['emotion_driver'].values
            
            # Get all targets at once
            y_train = train_df[self.targets].values
            y_test = test_df[self.targets].values
            
            # Fit vectorizer and transform data
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            
            # Train multitask classifier
            multi_classifier.fit(X_train_vec, y_train)
            y_pred = multi_classifier.predict(X_test_vec)
            y_pred_proba = multi_classifier.predict_proba(X_test_vec)
            
            # Calculate metrics for each target
            group_acc = []
            group_auc = []
            
            for i, target in enumerate(self.targets):
                acc = balanced_accuracy_score(y_test[:, i], y_pred[:, i])
                auc = roc_auc_score(y_test[:, i], y_pred_proba[i][:, 1])
                
                fold_scores[target]['acc'].append(acc)
                fold_scores[target]['auc'].append(auc)
                
                group_acc.append(acc)
                group_auc.append(auc)
                
                print(f"Group {group}, {target}: Acc={acc:.4f}, AUC={auc:.4f}")
            
            # Calculate combined metrics for this group
            combined_acc = np.mean(group_acc)
            combined_auc = np.mean(group_auc)
            combined_scores['acc'].append(combined_acc)
            combined_scores['auc'].append(combined_auc)
            print(f"Group {group}, COMBINED: Acc={combined_acc:.4f}, AUC={combined_auc:.4f}")
        
        # Store results
        self.results[model_name] = {
            'individual_targets': {},
            'combined': {
                'balanced_accuracy': (np.mean(combined_scores['acc']), np.std(combined_scores['acc'])),
                'auc': (np.mean(combined_scores['auc']), np.std(combined_scores['auc']))
            }
        }
        
        for target in self.targets:
            self.results[model_name]['individual_targets'][target] = {
                'balanced_accuracy': (np.mean(fold_scores[target]['acc']), np.std(fold_scores[target]['acc'])),
                'auc': (np.mean(fold_scores[target]['auc']), np.std(fold_scores[target]['auc']))
            }
        
        # Print summary
        print(f"\n{model_name} Multitask Results Summary:")
        for target in self.targets:
            acc_mean, acc_std = self.results[model_name]['individual_targets'][target]['balanced_accuracy']
            auc_mean, auc_std = self.results[model_name]['individual_targets'][target]['auc']
            print(f"{target}: Acc={acc_mean:.4f}±{acc_std:.4f}, AUC={auc_mean:.4f}±{auc_std:.4f}")
        
        # Print combined results
        combined_acc_mean, combined_acc_std = self.results[model_name]['combined']['balanced_accuracy']
        combined_auc_mean, combined_auc_std = self.results[model_name]['combined']['auc']
        print(f"COMBINED: Acc={combined_acc_mean:.4f}±{combined_acc_std:.4f}, AUC={combined_auc_mean:.4f}±{combined_auc_std:.4f}")
    
    def evaluate_vader(self):
        """
        Evaluate VADER sentiment analysis with multitask learning
        """
        print("Evaluating VADER with multitask approach...")
        analyzer = SentimentIntensityAnalyzer()
        fold_scores = {target: {'acc': [], 'auc': []} for target in self.targets}
        combined_scores = {'acc': [], 'auc': []}
        
        # Create a multi-output classifier
        multi_classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000), n_jobs=-1)
        
        for group in range(1, 6):  # 5 groups
            # Load data for this group
            train_df, test_df = self.load_group_data(group)
            
            # Extract features
            train_features = []
            for text in train_df['emotion_driver']:
                scores = analyzer.polarity_scores(text)
                train_features.append([scores['pos'], scores['neg'], scores['neu'], scores['compound']])
            
            test_features = []
            for text in test_df['emotion_driver']:
                scores = analyzer.polarity_scores(text)
                test_features.append([scores['pos'], scores['neg'], scores['neu'], scores['compound']])
            
            train_features = np.array(train_features)
            test_features = np.array(test_features)
            
            # Get all targets at once (ensure they are properly formatted)
            y_train = train_df[self.targets].values.astype(int)
            y_test = test_df[self.targets].values.astype(int)
            
            # Train multitask classifier
            multi_classifier.fit(train_features, y_train)
            y_pred = multi_classifier.predict(test_features)
            y_pred_proba = multi_classifier.predict_proba(test_features)
            
            # Calculate metrics for each target
            group_acc = []
            group_auc = []
            
            for i, target in enumerate(self.targets):
                acc = balanced_accuracy_score(y_test[:, i], y_pred[:, i])
                auc = roc_auc_score(y_test[:, i], y_pred_proba[i][:, 1])
                
                fold_scores[target]['acc'].append(acc)
                fold_scores[target]['auc'].append(auc)
                
                group_acc.append(acc)
                group_auc.append(auc)
                
                print(f"Group {group}, {target}: Acc={acc:.4f}, AUC={auc:.4f}")
            
            # Calculate combined metrics for this group
            combined_acc = np.mean(group_acc)
            combined_auc = np.mean(group_auc)
            combined_scores['acc'].append(combined_acc)
            combined_scores['auc'].append(combined_auc)
            print(f"Group {group}, COMBINED: Acc={combined_acc:.4f}, AUC={combined_auc:.4f}")
        
        # Store results
        self.results['VADER'] = {
            'individual_targets': {},
            'combined': {
                'balanced_accuracy': (np.mean(combined_scores['acc']), np.std(combined_scores['acc'])),
                'auc': (np.mean(combined_scores['auc']), np.std(combined_scores['auc']))
            }
        }
        
        for target in self.targets:
            self.results['VADER']['individual_targets'][target] = {
                'balanced_accuracy': (np.mean(fold_scores[target]['acc']), np.std(fold_scores[target]['acc'])),
                'auc': (np.mean(fold_scores[target]['auc']), np.std(fold_scores[target]['auc']))
            }
        
        # Print summary
        print("\nVADER Multitask Results Summary:")
        for target in self.targets:
            acc_mean, acc_std = self.results['VADER']['individual_targets'][target]['balanced_accuracy']
            auc_mean, auc_std = self.results['VADER']['individual_targets'][target]['auc']
            print(f"{target}: Acc={acc_mean:.4f}±{acc_std:.4f}, AUC={auc_mean:.4f}±{auc_std:.4f}")
        
        # Print combined results
        combined_acc_mean, combined_acc_std = self.results['VADER']['combined']['balanced_accuracy']
        combined_auc_mean, combined_auc_std = self.results['VADER']['combined']['auc']
        print(f"COMBINED: Acc={combined_acc_mean:.4f}±{combined_acc_std:.4f}, AUC={combined_auc_mean:.4f}±{combined_auc_std:.4f}")
    
    def evaluate_bert(self, params=None):
        """
        Evaluate BERT model with multitask approach using the provided parameters
        
        Args:
            params: Dictionary of hyperparameters. If None, default parameters are used.
        """
        # Default parameters if none provided
        if params is None:
            params = {
                'batch_size': 32,
                'max_seq_length': 128,
                'learning_rate': 3e-5,
                'epochs': 3,
                'classifier_dropout': 0.1
            }
        
        print("Loading BERT model for multitask learning...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
        print(f"BERT model running on device: {self.device}")
        print(f"Using parameters: {params}")
        
        fold_scores = {target: {'acc': [], 'auc': []} for target in self.targets}
        combined_scores = {'acc': [], 'auc': []}
        
        # Use batch size from parameters
        batch_size = params.get('batch_size', 32)
        max_seq_length = params.get('max_seq_length', 128)
        
        # Save checkpoint before starting
        self.save_checkpoint('BERT')
        
        for group in range(1, 6):
            # Check system resources
            if not self.check_system_resources():
                print("Waiting for system resources to stabilize...")
                time.sleep(30)
            
            # Load data for this group
            train_df, test_df = self.load_group_data(group)
            
            # Prepare data
            X_train = train_df['emotion_driver'].values
            X_test = test_df['emotion_driver'].values
            
            # Get all targets at once (ensure they are integers)
            y_train = train_df[self.targets].values.astype(int)
            y_test = test_df[self.targets].values.astype(int)
            
            # Get BERT embeddings with parameters
            print(f"Processing train embeddings for group {group}...")
            train_embeddings = self._create_embeddings_in_chunks(
                model, tokenizer, X_train,
                chunk_size=batch_size, 
                max_seq_length=max_seq_length
            )
            torch.cuda.empty_cache()  # Clear GPU memory after processing
            gc.collect()
            
            # Cooling period between processing
            time.sleep(3)
            
            print(f"Processing test embeddings for group {group}...")
            test_embeddings = self._create_embeddings_in_chunks(
                model, tokenizer, X_test,
                chunk_size=batch_size, 
                max_seq_length=max_seq_length
            )
            torch.cuda.empty_cache()  # Clear GPU memory after processing
            gc.collect()
            
            # Train multitask classifier
            print(f"Training multitask classifier for group {group}...")
            multi_classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000), n_jobs=-1)
            multi_classifier.fit(train_embeddings, y_train)
            y_pred = multi_classifier.predict(test_embeddings)
            y_pred_proba = multi_classifier.predict_proba(test_embeddings)
            
            # Calculate metrics for each target
            group_acc = []
            group_auc = []
            
            for i, target in enumerate(self.targets):
                acc = balanced_accuracy_score(y_test[:, i], y_pred[:, i])
                auc = roc_auc_score(y_test[:, i], y_pred_proba[i][:, 1])
                
                fold_scores[target]['acc'].append(acc)
                fold_scores[target]['auc'].append(auc)
                
                group_acc.append(acc)
                group_auc.append(auc)
                
                print(f"Group {group}, {target}: Acc={acc:.4f}, AUC={auc:.4f}")
            
            # Calculate combined metrics for this group
            combined_acc = np.mean(group_acc)
            combined_auc = np.mean(group_auc)
            combined_scores['acc'].append(combined_acc)
            combined_scores['auc'].append(combined_auc)
            print(f"Group {group}, COMBINED: Acc={combined_acc:.4f}, AUC={combined_auc:.4f}")
            
            # Force clean up
            del train_embeddings, test_embeddings, multi_classifier
            gc.collect()
            torch.cuda.empty_cache()
            
            # Cooling period between folds
            print(f"Cooling period after group {group}...")
            time.sleep(10)
        
        # Store results
        self.results['BERT'] = {
            'individual_targets': {},
            'combined': {
                'balanced_accuracy': (np.mean(combined_scores['acc']), np.std(combined_scores['acc'])),
                'auc': (np.mean(combined_scores['auc']), np.std(combined_scores['auc']))
            }
        }
        
        for target in self.targets:
            self.results['BERT']['individual_targets'][target] = {
                'balanced_accuracy': (np.mean(fold_scores[target]['acc']), np.std(fold_scores[target]['acc'])),
                'auc': (np.mean(fold_scores[target]['auc']), np.std(fold_scores[target]['auc']))
            }
        
        # Print summary
        print("\nBERT Multitask Results Summary:")
        for target in self.targets:
            acc_mean, acc_std = self.results['BERT']['individual_targets'][target]['balanced_accuracy']
            auc_mean, auc_std = self.results['BERT']['individual_targets'][target]['auc']
            print(f"{target}: Acc={acc_mean:.4f}±{acc_std:.4f}, AUC={auc_mean:.4f}±{auc_std:.4f}")
        
        # Print combined results
        combined_acc_mean, combined_acc_std = self.results['BERT']['combined']['balanced_accuracy']
        combined_auc_mean, combined_auc_std = self.results['BERT']['combined']['auc']
        print(f"COMBINED: Acc={combined_acc_mean:.4f}±{combined_acc_std:.4f}, AUC={combined_auc_mean:.4f}±{combined_auc_std:.4f}")
        
        # Save results
        self.save_model_results('BERT')
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(15)  # Extended cooling period
    
    def save_model_results(self, model_name):
        """Save results for a specific model"""
        if model_name in self.results:
            result_path = os.path.join(
                self.output_dir, 
                f"{model_name}_multitask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(result_path, 'w') as f:
                json.dump({model_name: self.results[model_name]}, f, indent=2)
            
            print(f"Results saved for {model_name}")
    
    def get_bert_embeddings(self, model, tokenizer, texts):
        """Helper function to get BERT embeddings with improved settings"""
        embeddings = []
        batch_size = self.batch_size
        
        # Process in chunks
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            # Check system resources less frequently with larger batches
            if i % (batch_size * 10) == 0 and i > 0:
                self.check_system_resources()
            
            batch_texts = texts[i:i + batch_size]
            encodings = tokenizer(
                batch_texts.tolist(), 
                truncation=True, 
                padding=True, 
                max_length=self.max_seq_length,
                return_tensors='pt'
            )
            
            # Move inputs to device
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            
            with torch.no_grad():
                outputs = model(**encodings)
                # Get embeddings from the output
                if hasattr(outputs, 'last_hidden_state'):
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                else:
                    batch_embeddings = outputs[0][:, 0, :].cpu().numpy()
                
                embeddings.append(batch_embeddings)
            
            # Clear references for garbage collection
            del encodings, outputs, batch_embeddings
            
            # Less frequent pauses between batches
            if i % (batch_size * 10) == 0 and i > 0:
                torch.cuda.empty_cache()
                time.sleep(0.2)  # Reduced from 0.5
        
        # Combine results
        combined = np.vstack(embeddings)
        
        # Clear list to free memory
        del embeddings
        gc.collect()
        
        return combined
    
    def get_transformer_embeddings(self, model, tokenizer, texts):
        """Generic function to get embeddings from transformer models with improved settings"""
        embeddings = []
        batch_size = self.batch_size
        
        # Process in chunks
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            # Check system resources less frequently
            if i % (batch_size * 10) == 0 and i > 0:
                self.check_system_resources()
            
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts.tolist(), 
                truncation=True, 
                padding=True, 
                max_length=self.max_seq_length,
                return_tensors='pt'
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Get embeddings from the output
                if hasattr(outputs, 'last_hidden_state'):
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                else:
                    batch_embeddings = outputs[0][:, 0, :].cpu().numpy()
                
                embeddings.append(batch_embeddings)
            
            # Clear references
            del inputs, outputs, batch_embeddings
            
            # Less frequent pauses between batches
            if i % (batch_size * 10) == 0 and i > 0:
                torch.cuda.empty_cache()
                time.sleep(0.2)  # Reduced pause time
        
        # Combine results
        combined = np.vstack(embeddings)
        
        # Clear list to free memory
        del embeddings
        gc.collect()
        
        return combined
    
    def evaluate_sentence_transformer(self):
        """
        Evaluate Sentence-BERT model with multitask approach
        """
        print("Loading Sentence-BERT model for multitask learning...")
        model = SentenceTransformer('all-mpnet-base-v2')
        
        # Configure SentenceTransformer to use specific GPU
        if self.use_cuda:
            model = model.to(self.device)
            print(f"SentenceBERT model running on GPU: {self.device}")
        
        # Reduce batch size specifically for SentenceBERT to avoid GPU memory issues
        sent_bert_batch_size = max(8, self.batch_size // 4)  # Reduced batch size
        print(f"Using reduced batch size for SentenceBERT: {sent_bert_batch_size}")
        
        fold_scores = {target: {'acc': [], 'auc': []} for target in self.targets}
        combined_scores = {'acc': [], 'auc': []}
        
        # Save checkpoint before starting
        self.save_checkpoint('SentenceBERT')
        
        for group in range(1, 6):
            # Check system resources before processing group
            if not self.check_system_resources():
                print("Waiting for system resources to stabilize...")
                time.sleep(30)
                
            print(f"\nProcessing group {group} for SentenceBERT...")
            train_df, test_df = self.load_group_data(group)
            
            # Get all targets at once
            y_train = train_df[self.targets].values
            y_test = test_df[self.targets].values
            
            # Process data in smaller chunks to avoid memory issues
            print(f"Encoding train data for group {group} in smaller chunks...")
            train_texts = train_df['emotion_driver'].values.tolist()
            
            # Process training data in chunks
            train_embeddings_chunks = []
            chunk_size = 100  # Process 100 samples at a time
            for i in range(0, len(train_texts), chunk_size):
                # Check system resources periodically
                if i > 0 and i % 500 == 0:
                    if not self.check_system_resources():
                        print("High resource usage detected, cooling down...")
                        time.sleep(10)
                        torch.cuda.empty_cache()
                    
                chunk_texts = train_texts[i:i+chunk_size]
                chunk_embeddings = model.encode(
                    chunk_texts,
                    batch_size=sent_bert_batch_size,
                    show_progress_bar=False
                )
                train_embeddings_chunks.append(chunk_embeddings)
                
                # Clear GPU cache after each chunk
                if self.use_cuda and i % 500 == 0:
                    torch.cuda.empty_cache()
                    
            # Combine all chunks
            train_embeddings = np.vstack(train_embeddings_chunks)
            del train_embeddings_chunks
            gc.collect()
            torch.cuda.empty_cache()
            
            # Process test data in chunks
            print(f"Encoding test data for group {group} in smaller chunks...")
            test_texts = test_df['emotion_driver'].values.tolist()
            test_embeddings_chunks = []
            for i in range(0, len(test_texts), chunk_size):
                chunk_texts = test_texts[i:i+chunk_size]
                chunk_embeddings = model.encode(
                    chunk_texts,
                    batch_size=sent_bert_batch_size,
                    show_progress_bar=False
                )
                test_embeddings_chunks.append(chunk_embeddings)
                
                # Clear GPU cache after each chunk
                if self.use_cuda and i % 500 == 0:
                    torch.cuda.empty_cache()
                    
            # Combine all chunks
            test_embeddings = np.vstack(test_embeddings_chunks)
            del test_embeddings_chunks
            gc.collect()
            torch.cuda.empty_cache()
            
            # Take a break after encoding to let system cool down
            print("Encoding complete. Taking a short break to cool down...")
            time.sleep(5)
            
            # Train multitask classifier
            print(f"Training classifier for group {group}...")
            multi_classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000), n_jobs=-1)
            multi_classifier.fit(train_embeddings, y_train)
            y_pred = multi_classifier.predict(test_embeddings)
            y_pred_proba = multi_classifier.predict_proba(test_embeddings)
            
            # Calculate metrics for each target
            group_acc = []
            group_auc = []
            
            for i, target in enumerate(self.targets):
                acc = balanced_accuracy_score(y_test[:, i], y_pred[:, i])
                auc = roc_auc_score(y_test[:, i], y_pred_proba[i][:, 1])
                
                fold_scores[target]['acc'].append(acc)
                fold_scores[target]['auc'].append(auc)
                
                group_acc.append(acc)
                group_auc.append(auc)
                
                print(f"Group {group}, {target}: Acc={acc:.4f}, AUC={auc:.4f}")
            
            # Calculate combined metrics for this group
            combined_acc = np.mean(group_acc)
            combined_auc = np.mean(group_auc)
            combined_scores['acc'].append(combined_acc)
            combined_scores['auc'].append(combined_auc)
            print(f"Group {group}, COMBINED: Acc={combined_acc:.4f}, AUC={combined_auc:.4f}")
            
            # Aggressive clean up
            del train_embeddings, test_embeddings, multi_classifier, y_pred, y_pred_proba
            gc.collect()
            torch.cuda.empty_cache()
            
            # Save intermediate checkpoint after each group
            self.save_checkpoint('SentenceBERT', f'group_{group}')
            
            # Extended cooling period between groups
            print(f"Finished group {group}. Cooling down...")
            time.sleep(15)
        
        # Store results
        self.results['SentenceBERT'] = {
            'individual_targets': {},
            'combined': {
                'balanced_accuracy': (np.mean(combined_scores['acc']), np.std(combined_scores['acc'])),
                'auc': (np.mean(combined_scores['auc']), np.std(combined_scores['auc']))
            }
        }
        
        for target in self.targets:
            self.results['SentenceBERT']['individual_targets'][target] = {
                'balanced_accuracy': (np.mean(fold_scores[target]['acc']), np.std(fold_scores[target]['acc'])),
                'auc': (np.mean(fold_scores[target]['auc']), np.std(fold_scores[target]['auc']))
            }
        
        # Print summary
        print("\nSentenceBERT Multitask Results Summary:")
        for target in self.targets:
            acc_mean, acc_std = self.results['SentenceBERT']['individual_targets'][target]['balanced_accuracy']
            auc_mean, auc_std = self.results['SentenceBERT']['individual_targets'][target]['auc']
            print(f"{target}: Acc={acc_mean:.4f}±{acc_std:.4f}, AUC={auc_mean:.4f}±{auc_std:.4f}")
        
        # Print combined results
        combined_acc_mean, combined_acc_std = self.results['SentenceBERT']['combined']['balanced_accuracy']
        combined_auc_mean, combined_auc_std = self.results['SentenceBERT']['combined']['auc']
        print(f"COMBINED: Acc={combined_acc_mean:.4f}±{combined_acc_std:.4f}, AUC={combined_auc_mean:.4f}±{combined_auc_std:.4f}")
        
        # Save results
        self.save_model_results('SentenceBERT')
        
        # Final cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(20)  # Extended cooling period
    
    def evaluate_distilbert(self):
        """
        Evaluate DistilBERT model
        """
        print("Loading DistilBERT model...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        
        # Enable multi-GPU if available
        if self.multi_gpu:
            model = nn.DataParallel(model, device_ids=self.gpu_devices)
            print("DistilBERT model running on multiple GPUs")
        
        for target in tqdm(self.targets, desc="Evaluating DistilBERT"):
            fold_scores_acc = []
            fold_scores_auc = []
            
            for group in range(1, 6):
                train_df, test_df = self.load_group_data(group)
                
                # Get embeddings
                train_embeddings = self.get_transformer_embeddings(model, tokenizer, train_df['emotion_driver'].values)
                test_embeddings = self.get_transformer_embeddings(model, tokenizer, test_df['emotion_driver'].values)
                
                # Train classifier
                classifier = LogisticRegression(max_iter=1000)
                classifier.fit(train_embeddings, train_df[target].values)
                y_pred = classifier.predict(test_embeddings)
                y_pred_proba = classifier.predict_proba(test_embeddings)[:, 1]
                
                # Calculate metrics
                acc = balanced_accuracy_score(test_df[target].values, y_pred)
                auc = roc_auc_score(test_df[target].values, y_pred_proba)
                
                fold_scores_acc.append(acc)
                fold_scores_auc.append(auc)
            
            self.results[target]['DistilBERT'] = {
                'balanced_accuracy': (np.mean(fold_scores_acc), np.std(fold_scores_acc)),
                'auc': (np.mean(fold_scores_auc), np.std(fold_scores_auc))
            }
    
    def evaluate_xlnet(self):
        """
        Evaluate XLNet model with improved robustness
        """
        print("Loading XLNet model...")
        try:
            # Use AutoTokenizer to avoid mismatch
            tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
            model = AutoModel.from_pretrained('xlnet-base-cased').to(self.device)
            
            # Enable multi-GPU if available
            if self.multi_gpu:
                model = nn.DataParallel(model, device_ids=self.gpu_devices)
                print("XLNet model running on multiple GPUs")
            
            # Save checkpoint before starting
            self.save_checkpoint('XLNet')
            
            fold_scores = {target: {'acc': [], 'auc': []} for target in self.targets}
            combined_scores = {'acc': [], 'auc': []}
            
            for group in range(1, 6):
                # Check system resources
                if not self.check_system_resources():
                    print("Waiting for system resources to stabilize...")
                    time.sleep(30)
                    
                print(f"\nProcessing group {group} for XLNet...")
                train_df, test_df = self.load_group_data(group)
                
                # Get all targets at once
                y_train = train_df[self.targets].values
                y_test = test_df[self.targets].values
                
                # Use smaller sequence length for XLNet
                xlnet_max_length = min(128, self.max_seq_length)
                print(f"Using reduced sequence length for XLNet: {xlnet_max_length}")
                
                # Process train embeddings in chunks
                print(f"Processing train embeddings for group {group} in chunks...")
                train_texts = train_df['emotion_driver'].values
                
                # Create embeddings in chunks to avoid memory issues
                train_embeddings_chunks = []
                chunk_size = 24  # Even smaller chunks for XLNet (memory intensive)
                for i in range(0, len(train_texts), chunk_size):
                    # Check resources more frequently
                    if i > 0 and i % 96 == 0:
                        if not self.check_system_resources():
                            print("High resource usage detected, cooling down...")
                            time.sleep(10)
                            torch.cuda.empty_cache()
                    
                    chunk_texts = train_texts[i:i+chunk_size]
                    
                    # Get embeddings for this chunk
                    encodings = tokenizer(
                        chunk_texts.tolist(), 
                        truncation=True, 
                        padding=True, 
                        max_length=xlnet_max_length,
                        return_tensors='pt'
                    )
                    
                    # Move to device
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    
                    with torch.no_grad():
                        outputs = model(**encodings)
                        # Get embeddings from the output
                        if hasattr(outputs, 'last_hidden_state'):
                            chunk_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        else:
                            chunk_embeddings = outputs[0][:, 0, :].cpu().numpy()
                        
                        train_embeddings_chunks.append(chunk_embeddings)
                    
                    # Clear memory immediately
                    del encodings, outputs, chunk_embeddings
                    if self.use_cuda:
                        torch.cuda.empty_cache()
                    
                    # Add small pause between chunks
                    time.sleep(0.5)
                
                # Combine chunks
                train_embeddings = np.vstack(train_embeddings_chunks)
                del train_embeddings_chunks
                gc.collect()
                torch.cuda.empty_cache()
                
                # Take a cooling break
                print("Train embeddings complete. Cooling down...")
                time.sleep(15)
                
                # Process test embeddings in chunks
                print(f"Processing test embeddings for group {group} in chunks...")
                test_texts = test_df['emotion_driver'].values
                
                # Create embeddings in chunks
                test_embeddings_chunks = []
                for i in range(0, len(test_texts), chunk_size):
                    chunk_texts = test_texts[i:i+chunk_size]
                    
                    # Get embeddings for this chunk
                    encodings = tokenizer(
                        chunk_texts.tolist(), 
                        truncation=True, 
                        padding=True, 
                        max_length=xlnet_max_length,
                        return_tensors='pt'
                    )
                    
                    # Move to device
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    
                    with torch.no_grad():
                        outputs = model(**encodings)
                        # Get embeddings from the output
                        if hasattr(outputs, 'last_hidden_state'):
                            chunk_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        else:
                            chunk_embeddings = outputs[0][:, 0, :].cpu().numpy()
                        
                        test_embeddings_chunks.append(chunk_embeddings)
                    
                    # Clear memory
                    del encodings, outputs, chunk_embeddings
                    if self.use_cuda:
                        torch.cuda.empty_cache()
                    
                    # Add small pause between chunks
                    time.sleep(0.5)
                
                # Combine chunks
                test_embeddings = np.vstack(test_embeddings_chunks)
                del test_embeddings_chunks
                gc.collect()
                torch.cuda.empty_cache()
                
                # Take a break after encoding to let system cool down
                print("Encoding complete. Taking a break to cool down...")
                time.sleep(15)
                
                # Train multitask classifier
                print(f"Training classifier for group {group}...")
                multi_classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000), n_jobs=-1)
                multi_classifier.fit(train_embeddings, y_train)
                y_pred = multi_classifier.predict(test_embeddings)
                y_pred_proba = multi_classifier.predict_proba(test_embeddings)
                
                # Calculate metrics for each target
                group_acc = []
                group_auc = []
                
                for i, target in enumerate(self.targets):
                    acc = balanced_accuracy_score(y_test[:, i], y_pred[:, i])
                    auc = roc_auc_score(y_test[:, i], y_pred_proba[i][:, 1])
                    
                    fold_scores[target]['acc'].append(acc)
                    fold_scores[target]['auc'].append(auc)
                    
                    group_acc.append(acc)
                    group_auc.append(auc)
                    
                    print(f"Group {group}, {target}: Acc={acc:.4f}, AUC={auc:.4f}")
                
                # Calculate combined metrics for this group
                combined_acc = np.mean(group_acc)
                combined_auc = np.mean(group_auc)
                combined_scores['acc'].append(combined_acc)
                combined_scores['auc'].append(combined_auc)
                print(f"Group {group}, COMBINED: Acc={combined_acc:.4f}, AUC={combined_auc:.4f}")
                
                # Aggressive clean up
                del train_embeddings, test_embeddings, multi_classifier, y_pred, y_pred_proba
                gc.collect()
                torch.cuda.empty_cache()
                
                # Save intermediate checkpoint after each group
                self.save_checkpoint('XLNet', f'group_{group}')
                
                # Extended cooling period between groups
                print(f"Finished group {group}. Cooling down...")
                time.sleep(25)
            
            # Store results
            self.results['XLNet'] = {
                'individual_targets': {},
                'combined': {
                    'balanced_accuracy': (np.mean(combined_scores['acc']), np.std(combined_scores['acc'])),
                    'auc': (np.mean(combined_scores['auc']), np.std(combined_scores['auc']))
                }
            }
            
            for target in self.targets:
                self.results['XLNet']['individual_targets'][target] = {
                    'balanced_accuracy': (np.mean(fold_scores[target]['acc']), np.std(fold_scores[target]['acc'])),
                    'auc': (np.mean(fold_scores[target]['auc']), np.std(fold_scores[target]['auc']))
                }
            
            # Print summary
            print("\nXLNet Multitask Results Summary:")
            for target in self.targets:
                acc_mean, acc_std = self.results['XLNet']['individual_targets'][target]['balanced_accuracy']
                auc_mean, auc_std = self.results['XLNet']['individual_targets'][target]['auc']
                print(f"{target}: Acc={acc_mean:.4f}±{acc_std:.4f}, AUC={auc_mean:.4f}±{auc_std:.4f}")
            
            # Print combined results
            combined_acc_mean, combined_acc_std = self.results['XLNet']['combined']['balanced_accuracy']
            combined_auc_mean, combined_auc_std = self.results['XLNet']['combined']['auc']
            print(f"COMBINED: Acc={combined_acc_mean:.4f}±{combined_acc_std:.4f}, AUC={combined_auc_mean:.4f}±{combined_auc_std:.4f}")
            
            # Save results
            self.save_model_results('XLNet')
            
        except Exception as e:
            print(f"Error in XLNet evaluation: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # Save error information
            error_log_path = os.path.join(self.output_dir, f'error_XLNet_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            with open(error_log_path, 'w') as f:
                f.write(f"Error evaluating XLNet: {str(e)}\n")
                f.write(traceback.format_exc())
        
        finally:
            # Final cleanup
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(30)  # Extended cooling period
    
    def evaluate_emotion_transformer(self):
        """
        Evaluate Emotion-Transformer (2021) with improved robustness
        """
        print("Loading Emotion-Transformer model...")
        try:
            # Using a sentence transformer fine-tuned on emotion data
            model = SentenceTransformer('joeddav/distilbert-base-uncased-go-emotions-student')
            
            # Configure SentenceTransformer to use specific GPU
            if self.use_cuda:
                model = model.to(self.device)
                print(f"Emotion-Transformer model running on GPU: {self.device}")
            
            # Save checkpoint before starting
            self.save_checkpoint('Emotion-Transformer')
            
            fold_scores = {target: {'acc': [], 'auc': []} for target in self.targets}
            combined_scores = {'acc': [], 'auc': []}
            
            for group in range(1, 6):
                # Check system resources before processing group
                if not self.check_system_resources():
                    print("Waiting for system resources to stabilize...")
                    time.sleep(30)
                    
                print(f"\nProcessing group {group} for Emotion-Transformer...")
                train_df, test_df = self.load_group_data(group)
                
                # Get all targets at once
                y_train = train_df[self.targets].values
                y_test = test_df[self.targets].values
                
                # Process data in smaller chunks to avoid memory issues
                print(f"Encoding train data for group {group} in smaller chunks...")
                train_texts = train_df['emotion_driver'].values.tolist()
                
                # Process training data in chunks
                train_embeddings_chunks = []
                chunk_size = 100  # Process 100 samples at a time
                for i in range(0, len(train_texts), chunk_size):
                    # Check system resources periodically
                    if i > 0 and i % 500 == 0:
                        if not self.check_system_resources():
                            print("High resource usage detected, cooling down...")
                            time.sleep(10)
                            torch.cuda.empty_cache()
                            
                    chunk_texts = train_texts[i:i+chunk_size]
                    # Use reduced batch size for embeddings
                    emotion_batch_size = max(8, self.batch_size // 4)
                    chunk_embeddings = model.encode(
                        chunk_texts,
                        batch_size=emotion_batch_size,
                        show_progress_bar=True
                    )
                    train_embeddings_chunks.append(chunk_embeddings)
                    
                    # Clear GPU cache after each chunk
                    if self.use_cuda and i % 500 == 0:
                        torch.cuda.empty_cache()
                        
                # Combine all chunks
                train_embeddings = np.vstack(train_embeddings_chunks)
                del train_embeddings_chunks
                gc.collect()
                torch.cuda.empty_cache()
                
                # Process test data in chunks
                print(f"Encoding test data for group {group} in smaller chunks...")
                test_texts = test_df['emotion_driver'].values.tolist()
                test_embeddings_chunks = []
                for i in range(0, len(test_texts), chunk_size):
                    chunk_texts = test_texts[i:i+chunk_size]
                    chunk_embeddings = model.encode(
                        chunk_texts,
                        batch_size=emotion_batch_size,
                        show_progress_bar=True
                    )
                    test_embeddings_chunks.append(chunk_embeddings)
                    
                    # Clear GPU cache after each chunk
                    if self.use_cuda and i % 500 == 0:
                        torch.cuda.empty_cache()
                        
                # Combine all chunks
                test_embeddings = np.vstack(test_embeddings_chunks)
                del test_embeddings_chunks
                gc.collect()
                torch.cuda.empty_cache()
                
                # Take a break after encoding to let system cool down
                print("Encoding complete. Taking a short break to cool down...")
                time.sleep(5)
                
                # Train multitask classifier
                print(f"Training classifier for group {group}...")
                multi_classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000), n_jobs=-1)
                multi_classifier.fit(train_embeddings, y_train)
                y_pred = multi_classifier.predict(test_embeddings)
                y_pred_proba = multi_classifier.predict_proba(test_embeddings)
                
                # Calculate metrics for each target
                group_acc = []
                group_auc = []
                
                for i, target in enumerate(self.targets):
                    acc = balanced_accuracy_score(y_test[:, i], y_pred[:, i])
                    auc = roc_auc_score(y_test[:, i], y_pred_proba[i][:, 1])
                    
                    fold_scores[target]['acc'].append(acc)
                    fold_scores[target]['auc'].append(auc)
                    
                    group_acc.append(acc)
                    group_auc.append(auc)
                    
                    print(f"Group {group}, {target}: Acc={acc:.4f}, AUC={auc:.4f}")
                
                # Calculate combined metrics for this group
                combined_acc = np.mean(group_acc)
                combined_auc = np.mean(group_auc)
                combined_scores['acc'].append(combined_acc)
                combined_scores['auc'].append(combined_auc)
                print(f"Group {group}, COMBINED: Acc={combined_acc:.4f}, AUC={combined_auc:.4f}")
                
                # Aggressive clean up
                del train_embeddings, test_embeddings, multi_classifier, y_pred, y_pred_proba
                gc.collect()
                torch.cuda.empty_cache()
                
                # Save intermediate checkpoint after each group
                self.save_checkpoint('Emotion-Transformer', f'group_{group}')
                
                # Extended cooling period between groups
                print(f"Finished group {group}. Cooling down...")
                time.sleep(15)
            
            # Store results
            self.results['Emotion-Transformer'] = {
                'individual_targets': {},
                'combined': {
                    'balanced_accuracy': (np.mean(combined_scores['acc']), np.std(combined_scores['acc'])),
                    'auc': (np.mean(combined_scores['auc']), np.std(combined_scores['auc']))
                }
            }
            
            for target in self.targets:
                self.results['Emotion-Transformer']['individual_targets'][target] = {
                    'balanced_accuracy': (np.mean(fold_scores[target]['acc']), np.std(fold_scores[target]['acc'])),
                    'auc': (np.mean(fold_scores[target]['auc']), np.std(fold_scores[target]['auc']))
                }
            
            # Print summary
            print("\nEmotion-Transformer Multitask Results Summary:")
            for target in self.targets:
                acc_mean, acc_std = self.results['Emotion-Transformer']['individual_targets'][target]['balanced_accuracy']
                auc_mean, auc_std = self.results['Emotion-Transformer']['individual_targets'][target]['auc']
                print(f"{target}: Acc={acc_mean:.4f}±{acc_std:.4f}, AUC={auc_mean:.4f}±{auc_std:.4f}")
            
            # Print combined results
            combined_acc_mean, combined_acc_std = self.results['Emotion-Transformer']['combined']['balanced_accuracy']
            combined_auc_mean, combined_auc_std = self.results['Emotion-Transformer']['combined']['auc']
            print(f"COMBINED: Acc={combined_acc_mean:.4f}±{combined_acc_std:.4f}, AUC={combined_auc_mean:.4f}±{combined_auc_std:.4f}")
            
            # Save results
            self.save_model_results('Emotion-Transformer')
            
        except Exception as e:
            print(f"Error in Emotion-Transformer evaluation: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # Save error information
            error_log_path = os.path.join(self.output_dir, f'error_Emotion-Transformer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            with open(error_log_path, 'w') as f:
                f.write(f"Error evaluating Emotion-Transformer: {str(e)}\n")
                f.write(traceback.format_exc())
        
        finally:
            # Final cleanup
            if 'model' in locals():
                del model
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(20)  # Extended cooling period
    
    def evaluate_roberta(self):
        """
        Evaluate RoBERTa model with improved robustness
        """
        print("Loading RoBERTa model...")
        try:
            # Use AutoTokenizer to avoid mismatch
            tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            model = AutoModel.from_pretrained('roberta-base').to(self.device)
            
            # Enable multi-GPU if available
            if self.multi_gpu:
                model = nn.DataParallel(model, device_ids=self.gpu_devices)
                print(f"RoBERTa model running on multiple GPUs: {self.gpu_devices}")
            
            # Save checkpoint before starting
            self.save_checkpoint('RoBERTa')
            
            fold_scores = {target: {'acc': [], 'auc': []} for target in self.targets}
            combined_scores = {'acc': [], 'auc': []}
            
            for group in range(1, 6):
                # Check system resources
                if not self.check_system_resources():
                    print("Waiting for system resources to stabilize...")
                    time.sleep(30)
                    
                print(f"\nProcessing group {group} for RoBERTa...")
                train_df, test_df = self.load_group_data(group)
                
                # Get all targets at once
                y_train = train_df[self.targets].values
                y_test = test_df[self.targets].values
                
                # Use smaller sequence length for RoBERTa
                roberta_max_length = min(128, self.max_seq_length)
                print(f"Using reduced sequence length for RoBERTa: {roberta_max_length}")
                
                # Process train embeddings in chunks
                print(f"Processing train embeddings for group {group} in chunks...")
                train_texts = train_df['emotion_driver'].values
                
                # Create embeddings in chunks to avoid memory issues
                train_embeddings_chunks = []
                chunk_size = 32  # Smaller chunks 
                for i in range(0, len(train_texts), chunk_size):
                    # Check resources more frequently
                    if i > 0 and i % 128 == 0:
                        if not self.check_system_resources():
                            print("High resource usage detected, cooling down...")
                            time.sleep(10)
                            torch.cuda.empty_cache()
                    
                    chunk_texts = train_texts[i:i+chunk_size]
                    
                    # Get embeddings for this chunk
                    encodings = tokenizer(
                        chunk_texts.tolist(), 
                        truncation=True, 
                        padding=True, 
                        max_length=roberta_max_length,
                        return_tensors='pt'
                    )
                    
                    # Move to device
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    
                    with torch.no_grad():
                        outputs = model(**encodings)
                        # Get embeddings from the output
                        if hasattr(outputs, 'last_hidden_state'):
                            chunk_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        else:
                            chunk_embeddings = outputs[0][:, 0, :].cpu().numpy()
                        
                        train_embeddings_chunks.append(chunk_embeddings)
                    
                    # Clear memory immediately
                    del encodings, outputs, chunk_embeddings
                    if self.use_cuda:
                        torch.cuda.empty_cache()
                    
                    # Add small pause between chunks
                    time.sleep(0.5)
                
                # Combine chunks
                train_embeddings = np.vstack(train_embeddings_chunks)
                del train_embeddings_chunks
                gc.collect()
                torch.cuda.empty_cache()
                
                # Take a cooling break
                print("Train embeddings complete. Cooling down...")
                time.sleep(10)
                
                # Process test embeddings in chunks
                print(f"Processing test embeddings for group {group} in chunks...")
                test_texts = test_df['emotion_driver'].values
                
                # Create embeddings in chunks
                test_embeddings_chunks = []
                for i in range(0, len(test_texts), chunk_size):
                    chunk_texts = test_texts[i:i+chunk_size]
                    
                    # Get embeddings for this chunk
                    encodings = tokenizer(
                        chunk_texts.tolist(), 
                        truncation=True, 
                        padding=True, 
                        max_length=roberta_max_length,
                        return_tensors='pt'
                    )
                    
                    # Move to device
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    
                    with torch.no_grad():
                        outputs = model(**encodings)
                        # Get embeddings from the output
                        if hasattr(outputs, 'last_hidden_state'):
                            chunk_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        else:
                            chunk_embeddings = outputs[0][:, 0, :].cpu().numpy()
                        
                        test_embeddings_chunks.append(chunk_embeddings)
                    
                    # Clear memory
                    del encodings, outputs, chunk_embeddings
                    if self.use_cuda:
                        torch.cuda.empty_cache()
                    
                    # Add small pause between chunks
                    time.sleep(0.5)
                
                # Combine chunks
                test_embeddings = np.vstack(test_embeddings_chunks)
                del test_embeddings_chunks
                gc.collect()
                torch.cuda.empty_cache()
                
                # Take a break after encoding to let system cool down
                print("Encoding complete. Taking a break to cool down...")
                time.sleep(10)
                
                # Train multitask classifier
                print(f"Training classifier for group {group}...")
                multi_classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000), n_jobs=-1)
                multi_classifier.fit(train_embeddings, y_train)
                y_pred = multi_classifier.predict(test_embeddings)
                y_pred_proba = multi_classifier.predict_proba(test_embeddings)
                
                # Calculate metrics for each target
                group_acc = []
                group_auc = []
                
                for i, target in enumerate(self.targets):
                    acc = balanced_accuracy_score(y_test[:, i], y_pred[:, i])
                    auc = roc_auc_score(y_test[:, i], y_pred_proba[i][:, 1])
                    
                    fold_scores[target]['acc'].append(acc)
                    fold_scores[target]['auc'].append(auc)
                    
                    group_acc.append(acc)
                    group_auc.append(auc)
                    
                    print(f"Group {group}, {target}: Acc={acc:.4f}, AUC={auc:.4f}")
                
                # Calculate combined metrics for this group
                combined_acc = np.mean(group_acc)
                combined_auc = np.mean(group_auc)
                combined_scores['acc'].append(combined_acc)
                combined_scores['auc'].append(combined_auc)
                print(f"Group {group}, COMBINED: Acc={combined_acc:.4f}, AUC={combined_auc:.4f}")
                
                # Aggressive clean up
                del train_embeddings, test_embeddings, multi_classifier, y_pred, y_pred_proba
                gc.collect()
                torch.cuda.empty_cache()
                
                # Save intermediate checkpoint after each group
                self.save_checkpoint('RoBERTa', f'group_{group}')
                
                # Extended cooling period between groups
                print(f"Finished group {group}. Cooling down...")
                time.sleep(20)
            
            # Store results
            self.results['RoBERTa'] = {
                'individual_targets': {},
                'combined': {
                    'balanced_accuracy': (np.mean(combined_scores['acc']), np.std(combined_scores['acc'])),
                    'auc': (np.mean(combined_scores['auc']), np.std(combined_scores['auc']))
                }
            }
            
            for target in self.targets:
                self.results['RoBERTa']['individual_targets'][target] = {
                    'balanced_accuracy': (np.mean(fold_scores[target]['acc']), np.std(fold_scores[target]['acc'])),
                    'auc': (np.mean(fold_scores[target]['auc']), np.std(fold_scores[target]['auc']))
                }
            
            # Print summary
            print("\nRoBERTa Multitask Results Summary:")
            for target in self.targets:
                acc_mean, acc_std = self.results['RoBERTa']['individual_targets'][target]['balanced_accuracy']
                auc_mean, auc_std = self.results['RoBERTa']['individual_targets'][target]['auc']
                print(f"{target}: Acc={acc_mean:.4f}±{acc_std:.4f}, AUC={auc_mean:.4f}±{auc_std:.4f}")
            
            # Print combined results
            combined_acc_mean, combined_acc_std = self.results['RoBERTa']['combined']['balanced_accuracy']
            combined_auc_mean, combined_auc_std = self.results['RoBERTa']['combined']['auc']
            print(f"COMBINED: Acc={combined_acc_mean:.4f}±{combined_acc_std:.4f}, AUC={combined_auc_mean:.4f}±{combined_auc_std:.4f}")
            
            # Save results
            self.save_model_results('RoBERTa')
            
        except Exception as e:
            print(f"Error in RoBERTa evaluation: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # Save error information
            error_log_path = os.path.join(self.output_dir, f'error_RoBERTa_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            with open(error_log_path, 'w') as f:
                f.write(f"Error evaluating RoBERTa: {str(e)}\n")
                f.write(traceback.format_exc())
        
        finally:
            # Final cleanup
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(25)  # Extended cooling period
    
    def evaluate_roberta_emotion(self):
        """
        Evaluate RoBERTa-based emotion classification (2021) with improved robustness
        """
        print("Loading RoBERTa-Emotion model...")
        try:
            # Use AutoTokenizer to avoid mismatch
            tokenizer = AutoTokenizer.from_pretrained('SamLowe/roberta-base-go_emotions')
            model = AutoModel.from_pretrained('SamLowe/roberta-base-go_emotions').to(self.device)
            
            # Enable multi-GPU if available
            if self.multi_gpu:
                model = nn.DataParallel(model, device_ids=self.gpu_devices)
                print("RoBERTa-Emotion model running on multiple GPUs")
            
            # Save checkpoint before starting
            self.save_checkpoint('RoBERTa-Emotion')
            
            fold_scores = {target: {'acc': [], 'auc': []} for target in self.targets} 
            combined_scores = {'acc': [], 'auc': []}
            
            for group in range(1, 6):
                # Check system resources
                if not self.check_system_resources():
                    print("Waiting for system resources to stabilize...")
                    time.sleep(30)
                    
                print(f"\nProcessing group {group} for RoBERTa-Emotion...")
                train_df, test_df = self.load_group_data(group)
                
                # Get all targets at once
                y_train = train_df[self.targets].values
                y_test = test_df[self.targets].values
                
                # Use smaller sequence length
                roberta_max_length = min(128, self.max_seq_length)
                print(f"Using reduced sequence length for RoBERTa-Emotion: {roberta_max_length}")
                
                # Process train embeddings in chunks
                print(f"Processing train embeddings for group {group} in chunks...")
                train_texts = train_df['emotion_driver'].values
                
                # Create embeddings in chunks to avoid memory issues
                train_embeddings_chunks = []
                chunk_size = 32  # Smaller chunks 
                for i in range(0, len(train_texts), chunk_size):
                    # Check resources more frequently
                    if i > 0 and i % 128 == 0:
                        if not self.check_system_resources():
                            print("High resource usage detected, cooling down...")
                            time.sleep(10)
                            torch.cuda.empty_cache()
                    
                    chunk_texts = train_texts[i:i+chunk_size]
                    
                    # Get embeddings for this chunk
                    encodings = tokenizer(
                        chunk_texts.tolist(), 
                        truncation=True, 
                        padding=True, 
                        max_length=roberta_max_length,
                        return_tensors='pt'
                    )
                    
                    # Move to device
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    
                    with torch.no_grad():
                        outputs = model(**encodings)
                        # Get embeddings from the output
                        if hasattr(outputs, 'last_hidden_state'):
                            chunk_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        else:
                            chunk_embeddings = outputs[0][:, 0, :].cpu().numpy()
                        
                        train_embeddings_chunks.append(chunk_embeddings)
                    
                    # Clear memory immediately
                    del encodings, outputs, chunk_embeddings
                    if self.use_cuda:
                        torch.cuda.empty_cache()
                    
                    # Add small pause between chunks
                    time.sleep(0.5)
                
                # Combine chunks
                train_embeddings = np.vstack(train_embeddings_chunks)
                del train_embeddings_chunks
                gc.collect()
                torch.cuda.empty_cache()
                
                # Take a cooling break
                print("Train embeddings complete. Cooling down...")
                time.sleep(10)
                
                # Process test embeddings in chunks
                print(f"Processing test embeddings for group {group} in chunks...")
                test_texts = test_df['emotion_driver'].values
                
                # Create embeddings in chunks
                test_embeddings_chunks = []
                for i in range(0, len(test_texts), chunk_size):
                    chunk_texts = test_texts[i:i+chunk_size]
                    
                    # Get embeddings for this chunk
                    encodings = tokenizer(
                        chunk_texts.tolist(), 
                        truncation=True, 
                        padding=True, 
                        max_length=roberta_max_length,
                        return_tensors='pt'
                    )
                    
                    # Move to device
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    
                    with torch.no_grad():
                        outputs = model(**encodings)
                        # Get embeddings from the output
                        if hasattr(outputs, 'last_hidden_state'):
                            chunk_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        else:
                            chunk_embeddings = outputs[0][:, 0, :].cpu().numpy()
                        
                        test_embeddings_chunks.append(chunk_embeddings)
                    
                    # Clear memory
                    del encodings, outputs, chunk_embeddings
                    if self.use_cuda:
                        torch.cuda.empty_cache()
                    
                    # Add small pause between chunks
                    time.sleep(0.5)
                
                # Combine chunks
                test_embeddings = np.vstack(test_embeddings_chunks)
                del test_embeddings_chunks
                gc.collect()
                torch.cuda.empty_cache()
                
                # Take a break after encoding to let system cool down
                print("Encoding complete. Taking a break to cool down...")
                time.sleep(10)
                
                # Train multitask classifier
                print(f"Training classifier for group {group}...")
                multi_classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000), n_jobs=-1)
                multi_classifier.fit(train_embeddings, y_train)
                y_pred = multi_classifier.predict(test_embeddings)
                y_pred_proba = multi_classifier.predict_proba(test_embeddings)
                
                # Calculate metrics for each target
                group_acc = []
                group_auc = []
                
                for i, target in enumerate(self.targets):
                    acc = balanced_accuracy_score(y_test[:, i], y_pred[:, i])
                    auc = roc_auc_score(y_test[:, i], y_pred_proba[i][:, 1])
                    
                    fold_scores[target]['acc'].append(acc)
                    fold_scores[target]['auc'].append(auc)
                    
                    group_acc.append(acc)
                    group_auc.append(auc)
                    
                    print(f"Group {group}, {target}: Acc={acc:.4f}, AUC={auc:.4f}")
                
                # Calculate combined metrics for this group
                combined_acc = np.mean(group_acc)
                combined_auc = np.mean(group_auc)
                combined_scores['acc'].append(combined_acc)
                combined_scores['auc'].append(combined_auc)
                print(f"Group {group}, COMBINED: Acc={combined_acc:.4f}, AUC={combined_auc:.4f}")
                
                # Aggressive clean up
                del train_embeddings, test_embeddings, multi_classifier, y_pred, y_pred_proba
                gc.collect()
                torch.cuda.empty_cache()
                
                # Save intermediate checkpoint after each group
                self.save_checkpoint('RoBERTa-Emotion', f'group_{group}')
                
                # Extended cooling period between groups
                print(f"Finished group {group}. Cooling down...")
                time.sleep(20)
            
            # Store results
            self.results['RoBERTa-Emotion'] = {
                'individual_targets': {},
                'combined': {
                    'balanced_accuracy': (np.mean(combined_scores['acc']), np.std(combined_scores['acc'])),
                    'auc': (np.mean(combined_scores['auc']), np.std(combined_scores['auc']))
                }
            }
            
            for target in self.targets:
                self.results['RoBERTa-Emotion']['individual_targets'][target] = {
                    'balanced_accuracy': (np.mean(fold_scores[target]['acc']), np.std(fold_scores[target]['acc'])),
                    'auc': (np.mean(fold_scores[target]['auc']), np.std(fold_scores[target]['auc']))
                }
            
            # Print summary
            print("\nRoBERTa-Emotion Multitask Results Summary:")
            for target in self.targets:
                acc_mean, acc_std = self.results['RoBERTa-Emotion']['individual_targets'][target]['balanced_accuracy']
                auc_mean, auc_std = self.results['RoBERTa-Emotion']['individual_targets'][target]['auc']
                print(f"{target}: Acc={acc_mean:.4f}±{acc_std:.4f}, AUC={auc_mean:.4f}±{auc_std:.4f}")
            
            # Print combined results
            combined_acc_mean, combined_acc_std = self.results['RoBERTa-Emotion']['combined']['balanced_accuracy']
            combined_auc_mean, combined_auc_std = self.results['RoBERTa-Emotion']['combined']['auc']
            print(f"COMBINED: Acc={combined_acc_mean:.4f}±{combined_acc_std:.4f}, AUC={combined_auc_mean:.4f}±{combined_auc_std:.4f}")
            
            # Save results
            self.save_model_results('RoBERTa-Emotion')
            
        except Exception as e:
            print(f"Error in RoBERTa-Emotion evaluation: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # Save error information
            error_log_path = os.path.join(self.output_dir, f'error_RoBERTa-Emotion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            with open(error_log_path, 'w') as f:
                f.write(f"Error evaluating RoBERTa-Emotion: {str(e)}\n")
                f.write(traceback.format_exc())
        
        finally:
            # Final cleanup
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(25)  # Extended cooling period
    
    def evaluate_emobert(self):
        """
        Evaluate EmoBERT model (2020) with improved robustness
        """
        print("Loading EmoBERT model...")
        try:
            # Use AutoTokenizer to avoid class mismatch warnings
            tokenizer = AutoTokenizer.from_pretrained('bhadresh-savani/bert-base-go-emotion')
            model = AutoModel.from_pretrained('bhadresh-savani/bert-base-go-emotion').to(self.device)
            
            # Enable multi-GPU if available
            if self.multi_gpu:
                model = nn.DataParallel(model, device_ids=self.gpu_devices)
                print("EmoBERT model running on multiple GPUs")
            
            # Save checkpoint before starting
            self.save_checkpoint('EmoBERT')
            
            fold_scores = {target: {'acc': [], 'auc': []} for target in self.targets}
            combined_scores = {'acc': [], 'auc': []}
            
            for group in range(1, 6):
                # Check system resources
                if not self.check_system_resources():
                    print("Waiting for system resources to stabilize...")
                    time.sleep(30)
                    
                print(f"\nProcessing group {group} for EmoBERT...")
                train_df, test_df = self.load_group_data(group)
                
                # Get all targets at once
                y_train = train_df[self.targets].values
                y_test = test_df[self.targets].values
                
                # Use smaller sequence length for EmoBERT too
                emobert_max_length = min(128, self.max_seq_length)
                print(f"Using reduced sequence length for EmoBERT: {emobert_max_length}")
                
                # Process train embeddings in chunks
                print(f"Processing train embeddings for group {group} in chunks...")
                train_texts = train_df['emotion_driver'].values
                
                # Create embeddings in chunks to avoid memory issues
                train_embeddings_chunks = []
                chunk_size = 32  # Smaller chunks
                for i in range(0, len(train_texts), chunk_size):
                    # Check resources periodically
                    if i > 0 and i % 128 == 0:
                        if not self.check_system_resources():
                            print("High resource usage detected, cooling down...")
                            time.sleep(10)
                            torch.cuda.empty_cache()
                    
                    chunk_texts = train_texts[i:i+chunk_size]
                    
                    # Get embeddings for this chunk
                    encodings = tokenizer(
                        chunk_texts.tolist(), 
                        truncation=True, 
                        padding=True, 
                        max_length=emobert_max_length,
                        return_tensors='pt'
                    )
                    
                    # Move to device
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    
                    with torch.no_grad():
                        outputs = model(**encodings)
                        # Get embeddings from the output
                        if hasattr(outputs, 'last_hidden_state'):
                            chunk_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        else:
                            chunk_embeddings = outputs[0][:, 0, :].cpu().numpy()
                        
                        train_embeddings_chunks.append(chunk_embeddings)
                    
                    # Clear memory
                    del encodings, outputs, chunk_embeddings
                    if self.use_cuda:
                        torch.cuda.empty_cache()
                    
                    # Add small pause between chunks
                    time.sleep(0.5)
                
                # Combine chunks
                train_embeddings = np.vstack(train_embeddings_chunks)
                del train_embeddings_chunks
                gc.collect()
                torch.cuda.empty_cache()
                
                # Take a short break
                print("Train embeddings complete. Cooling down...")
                time.sleep(10)
                
                # Process test embeddings in chunks
                print(f"Processing test embeddings for group {group} in chunks...")
                test_texts = test_df['emotion_driver'].values
                
                # Create embeddings in chunks
                test_embeddings_chunks = []
                for i in range(0, len(test_texts), chunk_size):
                    chunk_texts = test_texts[i:i+chunk_size]
                    
                    # Get embeddings for this chunk
                    encodings = tokenizer(
                        chunk_texts.tolist(), 
                        truncation=True, 
                        padding=True, 
                        max_length=emobert_max_length,
                        return_tensors='pt'
                    )
                    
                    # Move to device
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    
                    with torch.no_grad():
                        outputs = model(**encodings)
                        # Get embeddings from the output
                        if hasattr(outputs, 'last_hidden_state'):
                            chunk_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        else:
                            chunk_embeddings = outputs[0][:, 0, :].cpu().numpy()
                        
                        test_embeddings_chunks.append(chunk_embeddings)
                    
                    # Clear memory
                    del encodings, outputs, chunk_embeddings
                    if self.use_cuda:
                        torch.cuda.empty_cache()
                    
                    # Add small pause between chunks
                    time.sleep(0.5)
                
                # Combine chunks
                test_embeddings = np.vstack(test_embeddings_chunks)
                del test_embeddings_chunks
                gc.collect()
                torch.cuda.empty_cache()
                
                # Take a break after encoding to let system cool down
                print("Encoding complete. Taking a short break to cool down...")
                time.sleep(10)
                
                # Train multitask classifier
                print(f"Training classifier for group {group}...")
                multi_classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000), n_jobs=-1)
                multi_classifier.fit(train_embeddings, y_train)
                y_pred = multi_classifier.predict(test_embeddings)
                y_pred_proba = multi_classifier.predict_proba(test_embeddings)
                
                # Calculate metrics for each target
                group_acc = []
                group_auc = []
                
                for i, target in enumerate(self.targets):
                    acc = balanced_accuracy_score(y_test[:, i], y_pred[:, i])
                    auc = roc_auc_score(y_test[:, i], y_pred_proba[i][:, 1])
                    
                    fold_scores[target]['acc'].append(acc)
                    fold_scores[target]['auc'].append(auc)
                    
                    group_acc.append(acc)
                    group_auc.append(auc)
                    
                    print(f"Group {group}, {target}: Acc={acc:.4f}, AUC={auc:.4f}")
                
                # Calculate combined metrics for this group
                combined_acc = np.mean(group_acc)
                combined_auc = np.mean(group_auc)
                combined_scores['acc'].append(combined_acc)
                combined_scores['auc'].append(combined_auc)
                print(f"Group {group}, COMBINED: Acc={combined_acc:.4f}, AUC={combined_auc:.4f}")
                
                # Aggressive clean up
                del train_embeddings, test_embeddings, multi_classifier, y_pred, y_pred_proba
                gc.collect()
                torch.cuda.empty_cache()
                
                # Save intermediate checkpoint after each group
                self.save_checkpoint('EmoBERT', f'group_{group}')
                
                # Extended cooling period between groups
                print(f"Finished group {group}. Cooling down...")
                time.sleep(20)
            
            # Store results
            self.results['EmoBERT'] = {
                'individual_targets': {},
                'combined': {
                    'balanced_accuracy': (np.mean(combined_scores['acc']), np.std(combined_scores['acc'])),
                    'auc': (np.mean(combined_scores['auc']), np.std(combined_scores['auc']))
                }
            }
            
            for target in self.targets:
                self.results['EmoBERT']['individual_targets'][target] = {
                    'balanced_accuracy': (np.mean(fold_scores[target]['acc']), np.std(fold_scores[target]['acc'])),
                    'auc': (np.mean(fold_scores[target]['auc']), np.std(fold_scores[target]['auc']))
                }
            
            # Print summary
            print("\nEmoBERT Multitask Results Summary:")
            for target in self.targets:
                acc_mean, acc_std = self.results['EmoBERT']['individual_targets'][target]['balanced_accuracy']
                auc_mean, auc_std = self.results['EmoBERT']['individual_targets'][target]['auc']
                print(f"{target}: Acc={acc_mean:.4f}±{acc_std:.4f}, AUC={auc_mean:.4f}±{auc_std:.4f}")
            
            # Print combined results
            combined_acc_mean, combined_acc_std = self.results['EmoBERT']['combined']['balanced_accuracy']
            combined_auc_mean, combined_auc_std = self.results['EmoBERT']['combined']['auc']
            print(f"COMBINED: Acc={combined_acc_mean:.4f}±{combined_acc_std:.4f}, AUC={combined_auc_mean:.4f}±{combined_auc_std:.4f}")
            
            # Save results
            self.save_model_results('EmoBERT')
            
        except Exception as e:
            print(f"Error in EmoBERT evaluation: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # Save error information
            error_log_path = os.path.join(self.output_dir, f'error_EmoBERT_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            with open(error_log_path, 'w') as f:
                f.write(f"Error evaluating EmoBERT: {str(e)}\n")
                f.write(traceback.format_exc())
        
        finally:
            # Final cleanup
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(30)  # Extended cooling period
    
    def evaluate_deberta_emotion(self):
        """
        Evaluate DeBERTa for emotion analysis (2022) with improved robustness
        """
        print("Loading DeBERTa-Emotion model...")
        try:
            # Use auto tokenizer to avoid mismatches
            tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
            model = AutoModel.from_pretrained('microsoft/deberta-base').to(self.device)
            
            # Enable multi-GPU if available
            if self.multi_gpu:
                model = nn.DataParallel(model, device_ids=self.gpu_devices)
                print("DeBERTa-Emotion model running on multiple GPUs")
            
            # Save checkpoint before starting
            self.save_checkpoint('DeBERTa-Emotion')
            
            fold_scores = {target: {'acc': [], 'auc': []} for target in self.targets}
            combined_scores = {'acc': [], 'auc': []}
            
            for group in range(1, 6):
                # Check system resources
                if not self.check_system_resources():
                    print("Waiting for system resources to stabilize...")
                    time.sleep(30)
                    
                print(f"\nProcessing group {group} for DeBERTa-Emotion...")
                train_df, test_df = self.load_group_data(group)
                
                # Get all targets at once
                y_train = train_df[self.targets].values
                y_test = test_df[self.targets].values
                
                # Process train embeddings in very small chunks
                print(f"Processing train embeddings for group {group} in small chunks...")
                train_texts = train_df['emotion_driver'].values
                
                # Create embeddings in chunks to avoid memory issues
                train_embeddings_chunks = []
                chunk_size = 32  # Even smaller chunks for DeBERTa (more memory intensive)
                
                # Use smaller sequence length for DeBERTa
                deberta_max_length = min(128, self.max_seq_length)
                print(f"Using reduced sequence length for DeBERTa: {deberta_max_length}")
                
                for i in range(0, len(train_texts), chunk_size):
                    # Check resources more frequently
                    if i > 0 and i % 128 == 0:
                        if not self.check_system_resources():
                            print("High resource usage detected, cooling down...")
                            time.sleep(10)
                            torch.cuda.empty_cache()
                    
                    chunk_texts = train_texts[i:i+chunk_size]
                    
                    # Get embeddings for this chunk
                    encodings = tokenizer(
                        chunk_texts.tolist(), 
                        truncation=True, 
                        padding=True, 
                        max_length=deberta_max_length,
                        return_tensors='pt'
                    )
                    
                    # Move to device
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    
                    with torch.no_grad():
                        outputs = model(**encodings)
                        # Get embeddings from the output
                        if hasattr(outputs, 'last_hidden_state'):
                            chunk_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        else:
                            chunk_embeddings = outputs[0][:, 0, :].cpu().numpy()
                        
                        train_embeddings_chunks.append(chunk_embeddings)
                    
                    # Clear memory immediately after each chunk
                    del encodings, outputs, chunk_embeddings
                    if self.use_cuda:
                        torch.cuda.empty_cache()
                    
                    # Add small pause between chunks
                    time.sleep(0.5)
                
                # Combine chunks
                train_embeddings = np.vstack(train_embeddings_chunks)
                del train_embeddings_chunks
                gc.collect()
                torch.cuda.empty_cache()
                
                # Take a cooling break
                print("Train embeddings complete. Cooling down...")
                time.sleep(10)
                
                # Process test embeddings in chunks
                print(f"Processing test embeddings for group {group} in small chunks...")
                test_texts = test_df['emotion_driver'].values
                
                # Create embeddings in chunks
                test_embeddings_chunks = []
                for i in range(0, len(test_texts), chunk_size):
                    chunk_texts = test_texts[i:i+chunk_size]
                    
                    # Get embeddings for this chunk
                    encodings = tokenizer(
                        chunk_texts.tolist(), 
                        truncation=True, 
                        padding=True, 
                        max_length=deberta_max_length,
                        return_tensors='pt'
                    )
                    
                    # Move to device
                    encodings = {k: v.to(self.device) for k, v in encodings.items()}
                    
                    with torch.no_grad():
                        outputs = model(**encodings)
                        # Get embeddings from the output
                        if hasattr(outputs, 'last_hidden_state'):
                            chunk_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        else:
                            chunk_embeddings = outputs[0][:, 0, :].cpu().numpy()
                        
                        test_embeddings_chunks.append(chunk_embeddings)
                    
                    # Clear memory
                    del encodings, outputs, chunk_embeddings
                    if self.use_cuda:
                        torch.cuda.empty_cache()
                    
                    # Add small pause between chunks
                    time.sleep(0.5)
                
                # Combine chunks
                test_embeddings = np.vstack(test_embeddings_chunks)
                del test_embeddings_chunks
                gc.collect()
                torch.cuda.empty_cache()
                
                # Take a break after encoding to let system cool down
                print("Encoding complete. Taking a break to cool down...")
                time.sleep(10)
                
                # Train multitask classifier
                print(f"Training classifier for group {group}...")
                multi_classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000), n_jobs=-1)
                multi_classifier.fit(train_embeddings, y_train)
                y_pred = multi_classifier.predict(test_embeddings)
                y_pred_proba = multi_classifier.predict_proba(test_embeddings)
                
                # Calculate metrics for each target
                group_acc = []
                group_auc = []
                
                for i, target in enumerate(self.targets):
                    acc = balanced_accuracy_score(y_test[:, i], y_pred[:, i])
                    auc = roc_auc_score(y_test[:, i], y_pred_proba[i][:, 1])
                    
                    fold_scores[target]['acc'].append(acc)
                    fold_scores[target]['auc'].append(auc)
                    
                    group_acc.append(acc)
                    group_auc.append(auc)
                    
                    print(f"Group {group}, {target}: Acc={acc:.4f}, AUC={auc:.4f}")
                
                # Calculate combined metrics for this group
                combined_acc = np.mean(group_acc)
                combined_auc = np.mean(group_auc)
                combined_scores['acc'].append(combined_acc)
                combined_scores['auc'].append(combined_auc)
                print(f"Group {group}, COMBINED: Acc={combined_acc:.4f}, AUC={combined_auc:.4f}")
                
                # Aggressive clean up
                del train_embeddings, test_embeddings, multi_classifier, y_pred, y_pred_proba
                gc.collect()
                torch.cuda.empty_cache()
                
                # Save intermediate checkpoint after each group
                self.save_checkpoint('DeBERTa-Emotion', f'group_{group}')
                
                # Extended cooling period between groups
                print(f"Finished group {group}. Cooling down...")
                time.sleep(20)
            
            # Store results
            self.results['DeBERTa-Emotion'] = {
                'individual_targets': {},
                'combined': {
                    'balanced_accuracy': (np.mean(combined_scores['acc']), np.std(combined_scores['acc'])),
                    'auc': (np.mean(combined_scores['auc']), np.std(combined_scores['auc']))
                }
            }
            
            for target in self.targets:
                self.results['DeBERTa-Emotion']['individual_targets'][target] = {
                    'balanced_accuracy': (np.mean(fold_scores[target]['acc']), np.std(fold_scores[target]['acc'])),
                    'auc': (np.mean(fold_scores[target]['auc']), np.std(fold_scores[target]['auc']))
                }
            
            # Print summary
            print("\nDeBERTa-Emotion Multitask Results Summary:")
            for target in self.targets:
                acc_mean, acc_std = self.results['DeBERTa-Emotion']['individual_targets'][target]['balanced_accuracy']
                auc_mean, auc_std = self.results['DeBERTa-Emotion']['individual_targets'][target]['auc']
                print(f"{target}: Acc={acc_mean:.4f}±{acc_std:.4f}, AUC={auc_mean:.4f}±{auc_std:.4f}")
            
            # Print combined results
            combined_acc_mean, combined_acc_std = self.results['DeBERTa-Emotion']['combined']['balanced_accuracy']
            combined_auc_mean, combined_auc_std = self.results['DeBERTa-Emotion']['combined']['auc']
            print(f"COMBINED: Acc={combined_acc_mean:.4f}±{combined_acc_std:.4f}, AUC={combined_auc_mean:.4f}±{combined_auc_std:.4f}")
            
            # Save results
            self.save_model_results('DeBERTa-Emotion')
            
        except Exception as e:
            print(f"Error in DeBERTa-Emotion evaluation: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # Save error information
            error_log_path = os.path.join(self.output_dir, f'error_DeBERTa-Emotion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            with open(error_log_path, 'w') as f:
                f.write(f"Error evaluating DeBERTa-Emotion: {str(e)}\n")
                f.write(traceback.format_exc())
        
        finally:
            # Final cleanup
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(30)  # Extended cooling period
    
    def run_all_experiments(self, tune_hyperparams=False):
        """
        Run all baseline experiments with multitask approach and enhanced fault tolerance
        
        Args:
            tune_hyperparams: Whether to perform hyperparameter tuning before evaluation
        """
        # Define the models to evaluate
        models_to_evaluate = [
            ('BoW', lambda params: self.evaluate_traditional_ml(
                'BoW',
                CountVectorizer(**params['vectorizer']),
                LogisticRegression(**params['classifier'])
            )),
            ('TF-IDF', lambda params: self.evaluate_traditional_ml(
                'TF-IDF',
                TfidfVectorizer(**params['vectorizer']),
                LogisticRegression(**params['classifier'])
            )),
            ('VADER', self.evaluate_vader),  # No tuning needed for VADER
            ('BERT', lambda params: self.evaluate_bert(params)),
            ('DistilBERT', lambda params: self.evaluate_distilbert(params)),
            ('SentenceBERT', lambda params: self.evaluate_sentence_transformer(params)),
            ('EmoBERT', lambda params: self.evaluate_emobert(params)),
            ('RoBERTa', lambda params: self.evaluate_roberta(params)),
            ('RoBERTa-Emotion', lambda params: self.evaluate_roberta_emotion(params)),
            ('DeBERTa-Emotion', lambda params: self.evaluate_deberta_emotion(params)),
            ('XLNet', lambda params: self.evaluate_xlnet(params)),
            ('Emotion-Transformer', lambda params: self.evaluate_emotion_transformer(params))
        ]
        
        # Model configurations for hyperparameter tuning
        tuning_configs = {
            'BoW': {'class': CountVectorizer, 'classifier': LogisticRegression},
            'TF-IDF': {'class': TfidfVectorizer, 'classifier': LogisticRegression},
            'BERT': {'class': BertModel, 'tokenizer': BertTokenizer, 'path': 'bert-base-uncased'},
            'DistilBERT': {'class': DistilBertModel, 'tokenizer': DistilBertTokenizer, 'path': 'distilbert-base-uncased'},
            'EmoBERT': {'class': AutoModel, 'tokenizer': AutoTokenizer, 'path': 'bhadresh-savani/bert-base-go-emotion'},
            'RoBERTa': {'class': AutoModel, 'tokenizer': AutoTokenizer, 'path': 'roberta-base'},
            'RoBERTa-Emotion': {'class': AutoModel, 'tokenizer': AutoTokenizer, 'path': 'SamLowe/roberta-base-go_emotions'},
            'DeBERTa-Emotion': {'class': AutoModel, 'tokenizer': AutoTokenizer, 'path': 'microsoft/deberta-base'},
            'XLNet': {'class': AutoModel, 'tokenizer': AutoTokenizer, 'path': 'xlnet-base-cased'}
        }
        
        # Initialize completed models list for tracking progress
        self.completed_models = []
        
        # Initialize parameter storage
        self.tuned_params = {}
        
        # Check for existing checkpoint
        latest_checkpoint = self._find_latest_checkpoint()
        if latest_checkpoint:
            try:
                print(f"Found checkpoint: {latest_checkpoint}")
                with open(latest_checkpoint, 'r') as f:
                    checkpoint_data = json.load(f)
                
                # Restore state from checkpoint
                if 'results' in checkpoint_data:
                    self.results = checkpoint_data['results']
                    print("Restored previous results from checkpoint")
                
                if 'completed_models' in checkpoint_data:
                    self.completed_models = checkpoint_data['completed_models']
                    print(f"Resuming from checkpoint. Already completed: {', '.join(self.completed_models)}")
                
                if 'tuned_params' in checkpoint_data:
                    self.tuned_params = checkpoint_data['tuned_params']
                    print(f"Loaded previously tuned parameters")
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
                print("Starting from scratch")
        
        # Save an initial all-results file
        self._save_all_results('initial')
        
        # First phase: Hyperparameter tuning if requested
        if tune_hyperparams:
            print("\n=== Phase 1: Hyperparameter Tuning ===")
            
            # Tune traditional models
            for model_name in ['BoW', 'TF-IDF']:
                if model_name not in self.tuned_params:
                    print(f"\nTuning {model_name} hyperparameters...")
                    try:
                        params = self.tune_traditional_ml(
                            model_name, 
                            tuning_configs[model_name]['class'],
                            tuning_configs[model_name]['classifier']
                        )
                        self.tuned_params[model_name] = params
                        print(f"Tuning completed for {model_name}")
                    except Exception as e:
                        print(f"Error tuning {model_name}: {str(e)}")
                        self.tuned_params[model_name] = {
                            'vectorizer': {'min_df': 5, 'max_df': 0.95},
                            'classifier': {'max_iter': 1000, 'class_weight': 'balanced'}
                        }
                        print(f"Using default parameters for {model_name}")
            
            # Tune transformer models
            for model_name in ['BERT', 'DistilBERT', 'EmoBERT', 'RoBERTa', 'RoBERTa-Emotion', 'DeBERTa-Emotion', 'XLNet']:
                if model_name not in self.tuned_params:
                    print(f"\nTuning {model_name} hyperparameters...")
                    try:
                        params = self.tune_transformer_model(
                            model_name,
                            tuning_configs[model_name]['class'],
                            tuning_configs[model_name]['tokenizer'],
                            tuning_configs[model_name]['path']
                        )
                        self.tuned_params[model_name] = params
                        print(f"Tuning completed for {model_name}")
                    except Exception as e:
                        print(f"Error tuning {model_name}: {str(e)}")
                        self.tuned_params[model_name] = {
                            'batch_size': 32,
                            'learning_rate': 3e-5,
                            'max_seq_length': 128,
                            'epochs': 3,
                            'classifier_dropout': 0.1
                        }
                        print(f"Using default parameters for {model_name}")
                        
            # For SentenceBERT and Emotion-Transformer (SentenceTransformer-based)
            for model_name in ['SentenceBERT', 'Emotion-Transformer']:
                if model_name not in self.tuned_params:
                    self.tuned_params[model_name] = {
                        'batch_size': 16,  # Smaller batch size for these memory-intensive models
                        'max_seq_length': 128
                    }
                    print(f"Using default parameters for {model_name}")
            
            # Special case for VADER (no parameters)
            self.tuned_params['VADER'] = {}
            
            # Save tuned parameters in checkpoint
            self.save_checkpoint('tuning_complete')
            
            print("\nHyperparameter tuning completed for all models")
        
        # Second phase: Evaluation with tuned parameters
        print("\n=== Phase 2: Model Evaluation ===")
        
        # Load existing parameters if not tuned in this run
        if not tune_hyperparams and not self.tuned_params:
            # Set default parameters
            print("Using default parameters for all models")
            for model_name, _ in models_to_evaluate:
                if model_name in ['BoW', 'TF-IDF']:
                    self.tuned_params[model_name] = {
                        'vectorizer': {'min_df': 5, 'max_df': 0.95},
                        'classifier': {'max_iter': 1000, 'class_weight': 'balanced'}
                    }
                elif model_name in ['BERT', 'DistilBERT', 'EmoBERT', 'RoBERTa', 'RoBERTa-Emotion', 'DeBERTa-Emotion', 'XLNet']:
                    self.tuned_params[model_name] = {
                        'batch_size': 32,
                        'learning_rate': 3e-5,
                        'max_seq_length': 128,
                        'epochs': 3,
                        'classifier_dropout': 0.1
                    }
                elif model_name in ['SentenceBERT', 'Emotion-Transformer']:
                    self.tuned_params[model_name] = {
                        'batch_size': 16,
                        'max_seq_length': 128
                    }
                else:  # VADER
                    self.tuned_params[model_name] = {}
        
        # Run evaluation with tuned or default parameters
        for model_name, evaluate_func in models_to_evaluate:
            # Skip already completed models
            if model_name in self.completed_models:
                print(f"\nSkipping already completed model: {model_name}")
                continue
                
            try:
                print(f"\n{'='*50}")
                print(f"Starting multitask evaluation of {model_name}...")
                print(f"Using parameters: {self.tuned_params.get(model_name, 'No parameters')}")
                print(f"{'='*50}")
                
                # Save checkpoint before starting model
                self.save_checkpoint(model_name)
                
                # Run evaluation with parameters
                if model_name == 'VADER':
                    evaluate_func()  # VADER has no parameters
                else:
                    evaluate_func(self.tuned_params[model_name])
                
                # Mark model as completed and save checkpoint
                self.completed_models.append(model_name)
                self.save_checkpoint(model_name)
                
                # Save intermediate results for this model
                self._save_model_results(model_name)
                
                # Save all results so far
                self._save_all_results(f'after_{model_name}')
                
                # Force clean memory after each model
                torch.cuda.empty_cache()
                gc.collect()
                
                # Extended cooling period between models
                print(f"Extended cooling period after {model_name}...")
                time.sleep(30)
                
            except Exception as e:
                # Enhanced error logging
                print(f"\nERROR evaluating {model_name}: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print(traceback.format_exc())
                print(f"Skipping {model_name} and continuing with next model...")
                
                # Save error information
                error_log_path = os.path.join(self.output_dir, f'error_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
                with open(error_log_path, 'w') as f:
                    f.write(f"Error evaluating {model_name}: {str(e)}\n")
                    f.write(f"Error type: {type(e).__name__}\n")
                    f.write(traceback.format_exc())
                
                continue
        
        try:
            # Save final complete results
            print("\nSaving final complete multitask results...")
            final_results = self._save_all_results('final')
            
            print("\nMultitask evaluation complete!")
            
            return final_results
        except Exception as e:
            print(f"Error saving final results: {str(e)}")
            print(traceback.format_exc())
            
            # Try to return whatever results we have
            try:
                return self._create_summary_dataframe()
            except:
                return None
    
    def _find_latest_checkpoint(self):
        """Find the most recent checkpoint file"""
        checkpoint_files = [os.path.join(self.checkpoint_dir, f) for f in os.listdir(self.checkpoint_dir) 
                           if f.startswith('checkpoint_')]
        
        if not checkpoint_files:
            return None
            
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoint_files[0]
    
    def _save_model_results(self, model_name):
        """Save results for a specific model with timestamp"""
        if model_name in self.results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(
                self.output_dir, 
                f"{model_name}_multitask_{timestamp}.json"
            )
            
            with open(result_path, 'w') as f:
                json.dump({model_name: self.results[model_name]}, f, indent=2)
            
            print(f"Results saved for {model_name}")
    
    def _save_model_summary(self, model_name, timestamp):
        """Save a CSV summary for a single model"""
        if model_name in self.results:
            summary_data = []
            
            # Add individual target results
            if 'individual_targets' in self.results[model_name]:
                for target in self.targets:
                    if target in self.results[model_name]['individual_targets']:
                        acc_mean, acc_std = self.results[model_name]['individual_targets'][target]['balanced_accuracy']
                        auc_mean, auc_std = self.results[model_name]['individual_targets'][target]['auc']
                        
                        summary_data.append({
                            'Target': target,
                            'Model': model_name,
                            'Balanced_Accuracy': f"{acc_mean:.3f} (±{acc_std:.3f})",
                            'AUC': f"{auc_mean:.3f} (±{auc_std:.3f})"
                        })
            
            # Add combined results
            if 'combined' in self.results[model_name]:
                acc_mean, acc_std = self.results[model_name]['combined']['balanced_accuracy']
                auc_mean, auc_std = self.results[model_name]['combined']['auc']
                
                summary_data.append({
                    'Target': 'COMBINED',
                    'Model': model_name,
                    'Balanced_Accuracy': f"{acc_mean:.3f} (±{acc_std:.3f})",
                    'AUC': f"{auc_mean:.3f} (±{auc_std:.3f})"
                })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_path = os.path.join(self.output_dir, f'summary_{model_name}_{timestamp}.csv')
                summary_df.to_csv(summary_path, index=False)
                print(f"Summary saved for {model_name} at {summary_path}")
    
    def _save_all_results(self, stage):
        """Save all results accumulated so far with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save full JSON results
            all_results_path = os.path.join(self.output_dir, f'all_results_{stage}_{timestamp}.json')
            with open(all_results_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            print(f"All results saved to {all_results_path}")
            
            # Create and save comprehensive summary
            summary_df = self._create_summary_dataframe()
            
            summary_path = os.path.join(self.output_dir, f'summary_all_{stage}_{timestamp}.csv')
            summary_df.to_csv(summary_path, index=False)
            
            # Create pivot table format
            try:
                summary_pivot = summary_df.pivot_table(
                    index='Model', 
                    columns='Target', 
                    values=['Balanced_Accuracy', 'AUC'],
                    aggfunc='first'
                )
                
                pivot_path = os.path.join(self.output_dir, f'summary_pivot_{stage}_{timestamp}.csv')
                summary_pivot.to_csv(pivot_path)
                print(f"Pivot summary saved to {pivot_path}")
                
                return summary_pivot
            except Exception as e:
                print(f"Error creating pivot table: {str(e)}")
                return summary_df
                
        except Exception as e:
            print(f"Error saving all results: {str(e)}")
            return None
    
    def _create_summary_dataframe(self):
        """Create a summary dataframe from current results"""
        all_summary_data = []
        
        # Add individual target results
        for model_name in self.results:
            if 'individual_targets' in self.results[model_name]:
                for target in self.targets:
                    if target in self.results[model_name]['individual_targets']:
                        acc_mean, acc_std = self.results[model_name]['individual_targets'][target]['balanced_accuracy']
                        auc_mean, auc_std = self.results[model_name]['individual_targets'][target]['auc']
                        
                        all_summary_data.append({
                            'Target': target,
                            'Model': model_name,
                            'Balanced_Accuracy': f"{acc_mean:.3f} (±{acc_std:.3f})",
                            'AUC': f"{auc_mean:.3f} (±{auc_std:.3f})"
                        })
        
        # Add combined results
        for model_name in self.results:
            if 'combined' in self.results[model_name]:
                acc_mean, acc_std = self.results[model_name]['combined']['balanced_accuracy']
                auc_mean, auc_std = self.results[model_name]['combined']['auc']
                
                all_summary_data.append({
                    'Target': 'COMBINED',
                    'Model': model_name,
                    'Balanced_Accuracy': f"{acc_mean:.3f} (±{acc_std:.3f})",
                    'AUC': f"{auc_mean:.3f} (±{auc_std:.3f})"
                })
        
        return pd.DataFrame(all_summary_data)
    
    def tune_traditional_ml(self, model_name, vectorizer_class, classifier_class):
        """
        Tune hyperparameters for traditional ML models (BoW, TF-IDF) using GridSearchCV
        
        Args:
            model_name: Name of the model ('BoW' or 'TF-IDF')
            vectorizer_class: Class for vectorizer (CountVectorizer or TfidfVectorizer)
            classifier_class: Classifier class (LogisticRegression)
        """
        print(f"Starting hyperparameter tuning for {model_name}...")
        
        # Define parameters for vectorizer
        if model_name == 'BoW':
            vectorizer_params = {
                'min_df': [1, 2, 5],
                'max_df': [0.9, 0.95, 1.0],
                'ngram_range': [(1, 1), (1, 2)],
            }
        else:  # TF-IDF
            vectorizer_params = {
                'min_df': [1, 2, 5],
                'max_df': [0.9, 0.95, 1.0],
                'ngram_range': [(1, 1), (1, 2)],
                'norm': ['l1', 'l2'],
                'use_idf': [True, False],
            }
        
        # Define parameters for classifier
        classifier_params = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],  # liblinear supports both l1 and l2
            'class_weight': ['balanced', None],
        }
        
        # Store best parameters for each target
        self.best_params = {model_name: {'vectorizer': {}, 'classifier': {}}}
        
        # Tune for each target separately using nested cross-validation
        for target in self.targets:
            print(f"\nTuning {model_name} for target: {target}")
            best_accuracy = 0
            best_vectorizer_params = None
            best_classifier_params = None
            
            # Get all data from groups 1-5 for tuning with cross-validation
            all_data = []
            for group in range(1, 6):
                train_df, test_df = self.load_group_data(group)
                # Combine train and test for complete dataset
                group_df = pd.concat([train_df, test_df])
                all_data.append(group_df)
            
            all_df = pd.concat(all_data)
            texts = all_df['emotion_driver'].values
            labels = all_df[target].values
            
            # Use 5-fold cross-validation (matching the 5 groups)
            cv = 5
            
            # Grid search over vectorizer parameters
            for vect_params in self._param_grid_iterator(vectorizer_params):
                try:
                    # Initialize vectorizer with current parameters
                    vectorizer = vectorizer_class(**vect_params)
                    
                    # Transform all texts
                    features = vectorizer.fit_transform(texts)
                    
                    # Grid search over classifier parameters
                    for clf_params in self._param_grid_iterator(classifier_params):
                        try:
                            # Initialize classifier with current parameters
                            classifier = classifier_class(**clf_params)
                            
                            # Use cross-validation to evaluate
                            scores = []
                            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
                            for train_idx, test_idx in kf.split(features):
                                X_train, X_test = features[train_idx], features[test_idx]
                                y_train, y_test = labels[train_idx], labels[test_idx]
                                
                                classifier.fit(X_train, y_train)
                                y_pred = classifier.predict(X_test)
                                acc = balanced_accuracy_score(y_test, y_pred)
                                scores.append(acc)
                            
                            mean_acc = np.mean(scores)
                            
                            if mean_acc > best_accuracy:
                                best_accuracy = mean_acc
                                best_vectorizer_params = vect_params
                                best_classifier_params = clf_params
                                
                                print(f"New best for {target}: {mean_acc:.4f}")
                                print(f"Vectorizer: {vect_params}")
                                print(f"Classifier: {clf_params}")
                        except Exception as e:
                            print(f"Error with classifier params {clf_params}: {str(e)}")
                            continue
                except Exception as e:
                    print(f"Error with vectorizer params {vect_params}: {str(e)}")
                    continue
            
            print(f"\nBest parameters for {target} with {model_name}:")
            print(f"Best accuracy: {best_accuracy:.4f}")
            print(f"Best vectorizer parameters: {best_vectorizer_params}")
            print(f"Best classifier parameters: {best_classifier_params}")
            
            # Store best parameters
            self.best_params[model_name]['vectorizer'][target] = best_vectorizer_params
            self.best_params[model_name]['classifier'][target] = best_classifier_params
        
        # Save all best parameters
        params_path = os.path.join(self.output_dir, f"{model_name}_best_params.json")
        with open(params_path, 'w') as f:
            json.dump(self.best_params[model_name], f, indent=2)
        
        print(f"Best parameters saved to {params_path}")
        return self.best_params[model_name]
    
    def _param_grid_iterator(self, param_grid):
        """Helper method to iterate through all combinations of parameters"""
        keys = param_grid.keys()
        values = param_grid.values()
        for instance in itertools.product(*values):
            yield dict(zip(keys, instance))
    
    def tune_transformer_model(self, model_name, model_class, tokenizer_class, model_path):
        """
        Tune hyperparameters for transformer models using grouped cross-validation
        
        Args:
            model_name: Name of the model (e.g., 'BERT', 'RoBERTa')
            model_class: Class for the transformer model
            tokenizer_class: Class for the tokenizer
            model_path: Path to pre-trained model
        """
        print(f"Starting hyperparameter tuning for {model_name}...")
        
        # Define hyperparameter grid
        hp_grid = {
            'batch_size': [8, 16, 32],
            'learning_rate': [1e-5, 3e-5, 5e-5],
            'max_seq_length': [128, 256],
            'epochs': [2, 3, 4],
            'classifier_dropout': [0.1, 0.2, 0.3]
        }
        
        # Store best parameters for each target
        self.best_transformer_params = {model_name: {}}
        
        # Tune for combined multitask performance
        print(f"\nTuning {model_name} for multitask objectives")
        best_combined_accuracy = 0
        best_combined_params = None
        
        # To save memory, use a smaller subset of the hyperparameter combinations
        # focusing on the most important ones
        # Use first 3 groups for tuning, 2 groups for validation to be more efficient
        train_groups = [1, 2, 3]
        val_groups = [4, 5]
        
        # Get validation data
        val_dfs = []
        for group in val_groups:
            _, test_df = self.load_group_data(group)
            val_dfs.append(test_df)
        val_df = pd.concat(val_dfs)
        
        # Get training data
        train_dfs = []
        for group in train_groups:
            train_df, _ = self.load_group_data(group)
            train_dfs.append(train_df)
        train_df = pd.concat(train_dfs)
        
        # Key hyperparameters to tune
        reduced_hp_grid = {
            'batch_size': [16, 32],
            'learning_rate': [3e-5, 5e-5],
            'max_seq_length': [128],  # Use shorter sequences for efficiency
            'epochs': [3],
            'classifier_dropout': [0.1]
        }
        
        # Load tokenizer once
        tokenizer = tokenizer_class.from_pretrained(model_path)
        
        # Perform grid search
        for hp in self._param_grid_iterator(reduced_hp_grid):
            try:
                print(f"\nTrying parameters: {hp}")
                
                # Load model with current hyperparameters
                model = model_class.from_pretrained(model_path)
                if self.use_cuda:
                    model = model.to(self.device)
                
                # Process validation data upfront
                val_texts = val_df['emotion_driver'].values
                val_labels = val_df[self.targets].values
                
                # Create validation embeddings - using chunks for memory efficiency
                val_embeddings = self._create_embeddings_in_chunks(
                    model, tokenizer, val_texts, 
                    chunk_size=hp['batch_size'], 
                    max_seq_length=hp['max_seq_length']
                )
                
                # Process training data
                train_texts = train_df['emotion_driver'].values
                train_labels = train_df[self.targets].values
                
                # Create training embeddings
                train_embeddings = self._create_embeddings_in_chunks(
                    model, tokenizer, train_texts, 
                    chunk_size=hp['batch_size'], 
                    max_seq_length=hp['max_seq_length']
                )
                
                # Train multitask classifier
                multi_classifier = MultiOutputClassifier(
                    LogisticRegression(
                        max_iter=1000, 
                        C=1.0,  # Could also tune this parameter
                        class_weight='balanced'
                    ), 
                    n_jobs=-1
                )
                
                # Fit on training data
                multi_classifier.fit(train_embeddings, train_labels)
                
                # Predict on validation data
                val_preds = multi_classifier.predict(val_embeddings)
                val_preds_proba = multi_classifier.predict_proba(val_embeddings)
                
                # Calculate accuracies for all targets
                target_accs = []
                target_aucs = []
                
                for i, target in enumerate(self.targets):
                    acc = balanced_accuracy_score(val_labels[:, i], val_preds[:, i])
                    auc = roc_auc_score(val_labels[:, i], val_preds_proba[i][:, 1])
                    
                    target_accs.append(acc)
                    target_aucs.append(auc)
                    
                    print(f"{target}: Acc={acc:.4f}, AUC={auc:.4f}")
                
                # Calculate combined accuracy
                combined_acc = np.mean(target_accs)
                combined_auc = np.mean(target_aucs)
                
                print(f"Combined Accuracy: {combined_acc:.4f}, AUC: {combined_auc:.4f}")
                
                # Update best parameters if this is better
                if combined_acc > best_combined_accuracy:
                    best_combined_accuracy = combined_acc
                    best_combined_params = hp
                    print(f"New best combined accuracy: {combined_acc:.4f}")
                    print(f"Parameters: {hp}")
                
                # Clean up to free memory
                del model, train_embeddings, val_embeddings, multi_classifier
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(5)  # Let system cool down
                
            except Exception as e:
                print(f"Error with parameters {hp}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                continue
        
        # Save best parameters
        self.best_transformer_params[model_name] = best_combined_params
        
        # Save to file
        params_path = os.path.join(self.output_dir, f"{model_name}_best_params.json")
        with open(params_path, 'w') as f:
            json.dump({model_name: best_combined_params}, f, indent=2)
        
        print(f"\nBest parameters for {model_name}:")
        print(f"Best combined accuracy: {best_combined_accuracy:.4f}")
        print(f"Parameters: {best_combined_params}")
        print(f"Parameters saved to: {params_path}")
        
        return best_combined_params
    
    def _create_embeddings_in_chunks(self, model, tokenizer, texts, chunk_size=32, max_seq_length=128):
        """
        Create embeddings in chunks to manage memory better
        """
        embeddings_chunks = []
        
        for i in range(0, len(texts), chunk_size):
            chunk_texts = texts[i:i+chunk_size]
            
            # Create encodings
            encodings = tokenizer(
                chunk_texts.tolist(), 
                truncation=True, 
                padding=True, 
                max_length=max_seq_length,
                return_tensors='pt'
            )
            
            # Move to device
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            
            with torch.no_grad():
                outputs = model(**encodings)
                # Get embeddings from the output
                if hasattr(outputs, 'last_hidden_state'):
                    chunk_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                else:
                    chunk_embeddings = outputs[0][:, 0, :].cpu().numpy()
                
                embeddings_chunks.append(chunk_embeddings)
            
            # Clear memory
            del encodings, outputs
            if self.use_cuda:
                torch.cuda.empty_cache()
        
        # Combine all chunks
        combined_embeddings = np.vstack(embeddings_chunks)
        return combined_embeddings
    
    def generate_hyperparameter_tuning_report(self):
        """
        Generate a comprehensive report on the hyperparameter tuning results.
        This includes the best parameters for each model and their performance metrics.
        """
        # Check if we have tuned parameters
        if not hasattr(self, 'tuned_params') or not self.tuned_params:
            print("No tuned parameters found. Run hyperparameter tuning first.")
            return
        
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models": {}
        }
        
        # Add each model's parameters to the report
        for model_name, params in self.tuned_params.items():
            report["models"][model_name] = {
                "parameters": params,
            }
            
            # If we have validation results, add them
            if hasattr(self, 'validation_results') and model_name in self.validation_results:
                report["models"][model_name]["validation_metrics"] = self.validation_results[model_name]
        
        # Save report as JSON
        report_path = os.path.join(self.output_dir, f"hyperparameter_tuning_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also generate a markdown summary for easier reading
        md_report = f"# Hyperparameter Tuning Report\n\n"
        md_report += f"Generated on: {report['timestamp']}\n\n"
        
        for model_name, model_info in report["models"].items():
            md_report += f"## {model_name}\n\n"
            md_report += f"### Best Parameters\n\n"
            
            # Format parameters based on model type
            if model_name in ['BoW', 'TF-IDF']:
                md_report += f"#### Vectorizer Parameters\n\n"
                if 'vectorizer' in model_info['parameters']:
                    for param, value in model_info['parameters']['vectorizer'].items():
                        md_report += f"- {param}: {value}\n"
                
                md_report += f"\n#### Classifier Parameters\n\n"
                if 'classifier' in model_info['parameters']:
                    for param, value in model_info['parameters']['classifier'].items():
                        md_report += f"- {param}: {value}\n"
            else:
                # For transformer models
                for param, value in model_info['parameters'].items():
                    md_report += f"- {param}: {value}\n"
            
            md_report += "\n"
            
            # Add validation metrics if available
            if "validation_metrics" in model_info:
                md_report += f"### Validation Performance\n\n"
                md_report += f"- Mean Balanced Accuracy: {model_info['validation_metrics'].get('mean_balanced_accuracy', 'N/A')}\n"
                md_report += f"- Mean AUC: {model_info['validation_metrics'].get('mean_auc', 'N/A')}\n\n"
        
        # Save markdown report
        md_report_path = os.path.join(self.output_dir, f"hyperparameter_tuning_report.md")
        with open(md_report_path, 'w') as f:
            f.write(md_report)
        
        print(f"Hyperparameter tuning report saved to {report_path} and {md_report_path}")
        return report

if __name__ == "__main__":
    # Set paths
    data_dir = "/home/zhiyuan/Documents/cancer_survival/processed_file"
    output_dir = "/home/zhiyuan/Documents/cancer_survival/results/baseline"
    
    # Run experiments
    experiment = BaselineExperiment(data_dir, output_dir)
    summary_pivot = experiment.run_all_experiments()
    
    # Display results
    print("\nResults Summary:")
    print(summary_pivot.to_string())