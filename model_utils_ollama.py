"""
Model Utils - Updated to use Ollama with Mistral Nemo
Replaces transformers-based Mistral 7B with Ollama API
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import datetime
import json
import threading
from pathlib import Path

load_dotenv()

print("Loading machine troubleshooting system with Ollama...")

# Import Ollama model manager instead of transformers
try:
    from ollama_model import get_model
    OLLAMA_AVAILABLE = True
    print("  ✓ Ollama model manager loaded")
except ImportError:
    OLLAMA_AVAILABLE = False
    print("  ⚠ Ollama not available, using fallback mode")

# Rest of the DynamicKnowledgeBase class remains the same
class DynamicKnowledgeBase:
    """
    Dynamic knowledge base that learns from user contributions in real-time
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load initial data
        self.reference_df = self._load_initial_data()
        self.user_contributions_df = self._load_user_contributions()
        
        # Combine datasets
        self.combined_df = self._merge_datasets()
        
        # Create embeddings
        self.embeddings = self._create_embeddings()
        
        # Track metadata
        self.metadata = {
            'last_update': datetime.now().isoformat(),
            'total_entries': len(self.combined_df),
            'user_contributions': len(self.user_contributions_df),
            'embedding_version': 'all-MiniLM-L6-v2'
        }
        
        print(f"  ✓ Knowledge base initialized with {self.metadata['total_entries']} entries")
        print(f"  ✓ User contributions: {self.metadata['user_contributions']}")
    
    def _load_initial_data(self):
        """Load the original CSV dataset"""
        try:
            csv_file = os.getenv('DATABASE_PATH', 'mix_dataset_final.csv')
            df = pd.read_csv(csv_file).dropna(subset=['Machine Type', 'MACHINE', 'Problem Description'])
            df['source'] = 'original'
            df['date_added'] = datetime.now().isoformat()
            df['verified'] = True
            df['effectiveness_score'] = 1.0
            return self._ensure_string_columns(df)
        except Exception as e:
            print(f"Warning: Could not load reference data: {e}")
            return pd.DataFrame()
    
    def _load_user_contributions(self):
        """Load user-submitted solutions"""
        try:
            user_file = 'data/user_corrections.csv'
            if os.path.exists(user_file):
                df = pd.read_csv(user_file)
                
                # Rename columns to match original dataset structure
                column_mapping = {
                    'machine_type': 'Machine Type',
                    'machine_name': 'MACHINE',
                    'problem': 'Problem Description',
                    'root_cause': 'Root Cause',
                    'solution': 'Action Taken',
                    'timestamp': 'date_added'
                }
                df = df.rename(columns=column_mapping)
                
                # Add metadata
                df['source'] = 'user_contribution'
                df['verified'] = df.get('verified', False)
                df['effectiveness_score'] = df.get('effectiveness_score', 0.8)
                
                return self._ensure_string_columns(df)
        except Exception as e:
            print(f"Info: No user contributions yet: {e}")
        
        return pd.DataFrame()
    
    def _merge_datasets(self):
        """Merge original dataset with user contributions"""
        if self.user_contributions_df.empty:
            return self.reference_df.copy()
        
        # Ensure both dataframes have the same columns
        required_cols = ['Machine Type', 'MACHINE', 'Problem Description', 
                        'Root Cause', 'Action Taken', 'source', 'date_added', 
                        'verified', 'effectiveness_score']
        
        for col in required_cols:
            if col not in self.reference_df.columns:
                self.reference_df[col] = ''
            if col not in self.user_contributions_df.columns:
                self.user_contributions_df[col] = ''
        
        combined = pd.concat([self.reference_df, self.user_contributions_df], 
                            ignore_index=True)
        
        # Remove duplicates (keep most recent or highest effectiveness)
        combined = combined.sort_values(['effectiveness_score', 'date_added'], 
                                       ascending=False)
        combined = combined.drop_duplicates(
            subset=['Machine Type', 'MACHINE', 'Problem Description'], 
            keep='first'
        )
        
        return combined
    
    def _ensure_string_columns(self, df, columns=['Machine Type', 'MACHINE', 
                                                   'Problem Description', 'Root Cause', 
                                                   'Action Taken']):
        """Ensure all specified columns contain string values"""
        for col in columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
        return df
    
    def _create_embeddings(self):
        """Create embeddings for all entries"""
        if self.combined_df.empty:
            return np.array([])
        
        self.combined_df['combined'] = (
            self.combined_df['Machine Type'] + " " +
            self.combined_df['MACHINE'] + " " +
            self.combined_df['Problem Description']
        )
        
        print("Creating embeddings for knowledge base...")
        batch_size = 256
        num_batches = (len(self.combined_df) + batch_size - 1) // batch_size
        
        embeddings_list = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(self.combined_df))
            
            batch_texts = self.combined_df['combined'].iloc[start_idx:end_idx].tolist()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    batch_size=16
                )
            
            embeddings_list.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings_list)
        print(f"  ✓ Created embeddings matrix of shape {embeddings.shape}")
        return embeddings
    
    def add_user_solution(self, machine_type, machine_name, problem, 
                         root_cause, solution, verified=False, 
                         effectiveness_score=None, user_id=None):
        """Add a new user-submitted solution to the knowledge base"""
        with self.lock:
            try:
                # Create new entry
                new_entry = {
                    'Machine Type': machine_type,
                    'MACHINE': machine_name,
                    'Problem Description': problem,
                    'Root Cause': root_cause,
                    'Action Taken': solution,
                    'source': 'user_contribution',
                    'date_added': datetime.now().isoformat(),
                    'verified': verified,
                    'effectiveness_score': effectiveness_score or 0.8,
                    'user_id': user_id or 'anonymous'
                }
                
                # Check for duplicates
                duplicate = self._check_duplicate(new_entry)
                if duplicate:
                    return {
                        'success': False,
                        'message': 'Similar solution already exists',
                        'duplicate_id': duplicate
                    }
                
                # Add to dataframe
                new_df = pd.DataFrame([new_entry])
                self.combined_df = pd.concat([self.combined_df, new_df], 
                                            ignore_index=True)
                
                # Create embedding for new entry
                new_text = f"{machine_type} {machine_name} {problem}"
                new_embedding = self.embedding_model.encode([new_text])[0]
                
                # Add to embeddings matrix
                self.embeddings = np.vstack([self.embeddings, new_embedding])
                
                # Save to file
                self._save_user_contribution(new_entry)
                
                # Update metadata
                self.metadata['last_update'] = datetime.now().isoformat()
                self.metadata['total_entries'] = len(self.combined_df)
                self.metadata['user_contributions'] += 1
                
                print(f"  ✓ Added new user solution for {machine_name}")
                
                return {
                    'success': True,
                    'message': 'Solution added successfully',
                    'entry_id': len(self.combined_df) - 1
                }
                
            except Exception as e:
                print(f"Error adding user solution: {e}")
                return {
                    'success': False,
                    'message': f'Error: {str(e)}'
                }
    
    def _check_duplicate(self, new_entry):
        """Check if similar entry already exists"""
        if self.combined_df.empty:
            return None
        
        # Check for exact matches
        exact_match = self.combined_df[
            (self.combined_df['Machine Type'] == new_entry['Machine Type']) &
            (self.combined_df['MACHINE'] == new_entry['MACHINE']) &
            (self.combined_df['Problem Description'] == new_entry['Problem Description'])
        ]
        
        if not exact_match.empty:
            return exact_match.index[0]
        
        # Check for semantic similarity
        new_text = f"{new_entry['Machine Type']} {new_entry['MACHINE']} {new_entry['Problem Description']}"
        new_embedding = self.embedding_model.encode([new_text])[0]
        
        similarities = cosine_similarity([new_embedding], self.embeddings)[0]
        max_similarity = np.max(similarities)
        
        # If very similar (>95%), consider duplicate
        if max_similarity > 0.95:
            return np.argmax(similarities)
        
        return None
    
    def _save_user_contribution(self, entry):
        """Save user contribution to CSV file"""
        try:
            Path('data').mkdir(exist_ok=True)
            filepath = 'data/user_corrections.csv'
            
            # Convert to dataframe format for saving
            save_entry = {
                'machine_type': entry['Machine Type'],
                'machine_name': entry['MACHINE'],
                'problem': entry['Problem Description'],
                'root_cause': entry['Root Cause'],
                'solution': entry['Action Taken'],
                'timestamp': entry['date_added'],
                'verified': entry['verified'],
                'effectiveness_score': entry['effectiveness_score'],
                'user_id': entry.get('user_id', 'anonymous')
            }
            
            df = pd.DataFrame([save_entry])
            
            # Append to file or create new
            if os.path.exists(filepath):
                df.to_csv(filepath, mode='a', header=False, index=False)
            else:
                df.to_csv(filepath, index=False)
                
        except Exception as e:
            print(f"Error saving user contribution: {e}")
    
    def update_solution_effectiveness(self, entry_id, effectiveness_score, 
                                     user_feedback=None):
        """Update the effectiveness score of a solution based on user feedback"""
        with self.lock:
            try:
                if entry_id < 0 or entry_id >= len(self.combined_df):
                    return {'success': False, 'message': 'Invalid entry ID'}
                
                # Update effectiveness score (weighted average with previous)
                current_score = self.combined_df.loc[entry_id, 'effectiveness_score']
                new_score = (current_score + effectiveness_score) / 2
                
                self.combined_df.loc[entry_id, 'effectiveness_score'] = new_score
                self.combined_df.loc[entry_id, 'last_feedback'] = datetime.now().isoformat()
                
                # Log feedback
                if user_feedback:
                    self._log_feedback(entry_id, effectiveness_score, user_feedback)
                
                print(f"  ✓ Updated effectiveness score for entry {entry_id}: {new_score:.2f}")
                
                return {
                    'success': True,
                    'message': 'Effectiveness updated',
                    'new_score': new_score
                }
                
            except Exception as e:
                print(f"Error updating effectiveness: {e}")
                return {'success': False, 'message': str(e)}
    
    def _log_feedback(self, entry_id, score, feedback):
        """Log user feedback for analytics"""
        try:
            Path('data').mkdir(exist_ok=True)
            feedback_log = 'data/solution_feedback.json'
            
            feedback_entry = {
                'entry_id': int(entry_id),
                'score': float(score),
                'feedback': feedback,
                'timestamp': datetime.now().isoformat()
            }
            
            # Load existing feedback
            if os.path.exists(feedback_log):
                with open(feedback_log, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(feedback_entry)
            
            # Save updated feedback
            with open(feedback_log, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"Error logging feedback: {e}")
    
    def search(self, machine_type, machine_name, problem, threshold=0.60, 
               top_n=3, prefer_verified=True):
        """Search knowledge base with weighting for user contributions"""
        if self.combined_df.empty:
            return {'primary': [], 'additional': []}
        
        # Enhanced query
        query_text = f"{machine_type} {machine_name} {problem}"
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        results = []
        for i, score in enumerate(similarities):
            if score >= threshold * 0.75:  # Lower threshold for initial filtering
                row = self.combined_df.iloc[i]
                
                # Apply effectiveness weighting
                weighted_score = score * row.get('effectiveness_score', 1.0)
                
                # Boost verified solutions
                if prefer_verified and row.get('verified', False):
                    weighted_score *= 1.15
                
                # Boost recent user contributions (temporal relevance)
                if row.get('source') == 'user_contribution':
                    try:
                        date_added = datetime.fromisoformat(row['date_added'])
                        days_old = (datetime.now() - date_added).days
                        recency_boost = max(1.0, 1.2 - (days_old / 365))  # Decay over year
                        weighted_score *= recency_boost
                    except:
                        pass
                
                machine_info = {
                    'machine_type': row['Machine Type'],
                    'machine_name': row['MACHINE'],
                    'problem': row['Problem Description'],
                    'source': row.get('source', 'original'),
                    'verified': row.get('verified', True),
                    'effectiveness_score': row.get('effectiveness_score', 1.0),
                    'entry_id': i
                }
                
                result_text = f"Root Cause: {row['Root Cause']}. Action Taken: {row['Action Taken']}"
                results.append((result_text, float(weighted_score), machine_info))
        
        # Sort by weighted score
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Split into primary and additional
        primary_results = results[:top_n]
        additional_results = results[top_n:top_n+10]
        
        # Format additional results for dropdown
        dropdown_candidates = []
        for i, (text, score, machine_info) in enumerate(additional_results):
            condensed_problem = machine_info.get('problem', '')
            if len(condensed_problem) > 40:
                condensed_problem = condensed_problem[:37] + "..."
            
            dropdown_item = {
                'id': machine_info['entry_id'],
                'machine_name': machine_info.get('machine_name', ''),
                'condensed_problem': condensed_problem,
                'problem': machine_info.get('problem', ''),
                'confidence': int(score * 100),
                'source': machine_info.get('source', 'original'),
                'verified': machine_info.get('verified', True),
                'full_result': (text, score, machine_info)
            }
            dropdown_candidates.append(dropdown_item)
        
        print(f"Found {len(primary_results)} primary results and {len(dropdown_candidates)} additional candidates")
        
        return {
            'primary': primary_results,
            'additional': dropdown_candidates
        }
    
    def get_statistics(self):
        """Get knowledge base statistics"""
        stats = {
            'total_entries': len(self.combined_df),
            'original_entries': len(self.combined_df[self.combined_df['source'] == 'original']),
            'user_contributions': len(self.combined_df[self.combined_df['source'] == 'user_contribution']),
            'verified_solutions': len(self.combined_df[self.combined_df['verified'] == True]),
            'avg_effectiveness': float(self.combined_df['effectiveness_score'].mean()),
            'last_update': self.metadata['last_update'],
            'machine_types': self.combined_df['Machine Type'].nunique(),
            'unique_machines': self.combined_df['MACHINE'].nunique()
        }
        return stats


# Global knowledge base instance
_knowledge_base = None

def get_knowledge_base() -> DynamicKnowledgeBase:
    """Get or create global knowledge base instance"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = DynamicKnowledgeBase()
    return _knowledge_base


def find_in_dataset(machine_type, machine_name, problem, threshold=0.60, 
                   top_n=3, extra_results=15):
    """Enhanced search using dynamic knowledge base"""
    kb = get_knowledge_base()
    return kb.search(machine_type, machine_name, problem, threshold, top_n)


def add_user_solution(machine_type, machine_name, problem, root_cause, 
                     solution, verified=False, effectiveness_score=None, 
                     user_id=None):
    """Add user-submitted solution to knowledge base"""
    kb = get_knowledge_base()
    return kb.add_user_solution(
        machine_type, machine_name, problem, root_cause, solution,
        verified, effectiveness_score, user_id
    )


def update_solution_effectiveness(entry_id, effectiveness_score, user_feedback=None):
    """Update solution effectiveness based on user feedback"""
    kb = get_knowledge_base()
    return kb.update_solution_effectiveness(entry_id, effectiveness_score, user_feedback)


def generate_ai_response(machine_type, machine_name, problem_description, 
                        context_results=None):
    """
    Generate response using Ollama Mistral Nemo model with context from knowledge base
    """
    if not OLLAMA_AVAILABLE:
        return _fallback_response(machine_type, machine_name, problem_description)
    
    try:
        # Get Ollama model instance
        model = get_model()
        
        # Prepare context examples with source information
        context_examples = []
        if context_results:
            for result_text, score, machine_info in context_results:
                parts = result_text.split('Action Taken:')
                root_cause = parts[0].replace('Root Cause:', '').strip() if len(parts) > 0 else 'Unknown'
                action = parts[1].strip() if len(parts) > 1 else 'Unknown'
                
                context_examples.append({
                    'machine_name': machine_info.get('machine_name', 'Unknown'),
                    'problem': machine_info.get('problem', 'Unknown'),
                    'root_cause': root_cause,
                    'action': action,
                    'confidence': score,
                    'source': machine_info.get('source', 'database'),
                    'verified': machine_info.get('verified', True)
                })
        
        # Generate response using Ollama
        response = model.generate_troubleshooting_response(
            machine_type=machine_type,
            machine_name=machine_name,
            problem=problem_description,
            context_examples=context_examples if context_examples else None,
            max_tokens=512
        )
        
        return response
        
    except Exception as e:
        print(f"Error generating AI response: {str(e)}")
        return _fallback_response(machine_type, machine_name, problem_description)


def _fallback_response(machine_type, machine_name, problem_description):
    """Fallback response when Ollama is unavailable"""
    return (
        f"Root Cause: The {problem_description} in {machine_name} ({machine_type}) "
        f"typically indicates a malfunction in either the control system or mechanical components. "
        f"Possible Solutions: 1) Power cycle the machine completely. "
        f"2) Check for error codes on the control panel. "
        f"3) Inspect mechanical components related to the {problem_description}. "
        f"4) Verify all connections are secure. "
        f"5) Consult technical manual section for {machine_type} troubleshooting."
    )


def get_kb_statistics():
    """Get knowledge base statistics"""
    kb = get_knowledge_base()
    return kb.get_statistics()


print("✓ Enhanced machine troubleshooting system loaded successfully with Ollama!")
