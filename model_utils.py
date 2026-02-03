import torch
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

print("Loading machine troubleshooting system...")

# Import Mixtral model manager
try:
    from advanced_model import get_model
    MIXTRAL_AVAILABLE = True
    print("  ✓ Mixtral model manager loaded")
except ImportError:
    MIXTRAL_AVAILABLE = False
    print("  ⚠ Mixtral not available, using fallback mode")

def load_data():
    try:
        csv_file = os.getenv('DATABASE_PATH', 'mix_dataset_final.csv')
        df = pd.read_csv(csv_file).dropna(subset=['Machine Type', 'MACHINE', 'Problem Description'])
        return df
    except Exception as e:
        print(f"Warning: Could not load reference data: {e}")
        return None

reference_df = load_data()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
reference_embeddings = None

def ensure_string_columns(df, columns=['Machine Type', 'MACHINE', 'Problem Description', 'Root Cause', 'Action Taken']):
    """Ensure all specified columns contain string values"""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
    return df

reference_df = ensure_string_columns(reference_df)

if reference_df is not None:
    reference_df['combined'] = (
        reference_df['Machine Type'] + " " +
        reference_df['MACHINE'] + " " +
        reference_df['Problem Description']
    )

    print("Creating embeddings for reference data in batches...")
    batch_size = 256
    num_batches = (len(reference_df) + batch_size - 1) // batch_size

    reference_embeddings_list = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(reference_df))

        print(f"  Processing batch {i+1}/{num_batches} (samples {start_idx} to {end_idx})")
        batch_texts = reference_df['combined'].iloc[start_idx:end_idx].tolist()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch_embeddings = embedding_model.encode(
                batch_texts,
                show_progress_bar=False,
                batch_size=16
            )

        reference_embeddings_list.append(batch_embeddings)
        import gc
        gc.collect()

    reference_embeddings = np.vstack(reference_embeddings_list)
    print(f"  ✓ Created embeddings matrix of shape {reference_embeddings.shape}")

def find_in_dataset(machine_type, machine_name, problem, threshold=0.60, top_n=3, extra_results=15):
    """
    Enhanced search that returns both primary results and additional candidates
    UNCHANGED - keeps your existing semantic search logic
    """
    if reference_df is None:
        return {'primary': [], 'additional': []}

    print(f"Running search for: {machine_type} - {machine_name} - {problem}")

    keyword_results = keyword_search(reference_df, problem, machine_name, top_n=15)
    vector_results = vector_search(reference_df, problem, machine_name, top_n=15, threshold=threshold*0.75)

    combined_results = []
    seen_solutions = set()
    seen_machines = set()
    solution_to_machines = {}

    for source_results in [vector_results, keyword_results]:
        for text, score, machine_info in source_results:
            solution_fingerprint = text.lower().strip()
            machine_name_value = machine_info.get('machine_name', '').lower()

            if solution_fingerprint not in solution_to_machines:
                solution_to_machines[solution_fingerprint] = []
            solution_to_machines[solution_fingerprint].append((machine_name_value, score))

            if solution_fingerprint not in seen_solutions:
                combined_results.append((text, score, machine_info))
                seen_solutions.add(solution_fingerprint)

    combined_results.sort(key=lambda x: x[1], reverse=True)

    primary_results = []
    result_count = 0
    included_indices = set()

    for i, (text, score, machine_info) in enumerate(combined_results):
        if result_count >= top_n:
            break

        primary_results.append((text, score, machine_info))
        included_indices.add(i)
        result_count += 1

    dropdown_candidates = []
    problem_keywords = set(problem.lower().split())

    filtered_for_dropdown = []
    for i, (text, score, machine_info) in enumerate(combined_results):
        if i in included_indices:
            continue

        problem_desc = machine_info.get('problem', '').lower()
        matching_keywords = problem_keywords.intersection(set(problem_desc.split()))

        if len(matching_keywords) > 0 or score > threshold + 0.1:
            filtered_for_dropdown.append((i, text, score, machine_info, len(matching_keywords)))

    filtered_for_dropdown.sort(key=lambda x: (x[2], x[4]), reverse=True)
    filtered_for_dropdown = filtered_for_dropdown[:min(7, extra_results)]

    for i, text, score, machine_info, _ in filtered_for_dropdown:
        condensed_problem = machine_info.get('problem', '')
        if len(condensed_problem) > 40:
            condensed_problem = condensed_problem[:37] + "..."

        dropdown_item = {
            'id': i,
            'machine_name': machine_info.get('machine_name', ''),
            'condensed_problem': condensed_problem,
            'problem': machine_info.get('problem', ''),
            'confidence': int(score * 100),
            'full_result': (text, score, machine_info)
        }
        dropdown_candidates.append(dropdown_item)

    print(f"Found {len(primary_results)} primary results and {len(dropdown_candidates)} dropdown candidates")
    return {
        'primary': primary_results,
        'additional': dropdown_candidates
    }

def keyword_search(df, problem, machine_name, top_n=5):
    """Search using keyword matching logic - UNCHANGED"""
    problem_lower = problem.lower()
    machine_name_lower = machine_name.lower()

    results = []

    for idx, row in df.iterrows():
        row_problem = str(row['Problem Description']).lower()
        row_machine = str(row['MACHINE']).lower()
        row_cause = str(row['Root Cause']).lower() if 'Root Cause' in row else ""

        problem_words = set(problem_lower.split())
        row_words = set(row_problem.split())
        matching_words = len(problem_words.intersection(row_words))
        total_words = len(problem_words)
        word_ratio = matching_words / total_words if total_words > 0 else 0

        exact_match_score = 0
        if len(problem_lower) > 3 and problem_lower in row_problem:
            exact_match_score = 0.4

        machine_match = 0
        if machine_name_lower in row_machine:
            machine_match = 0.2

        key_terms = ["alarm", "error", "not working", "failed", "broken", "issue", "faulty", "malfunction"]
        term_match = 0
        for term in key_terms:
            if term in problem_lower and term in row_problem:
                term_match = 0.15
                break

        final_score = word_ratio * 0.4 + exact_match_score + machine_match + term_match

        if final_score > 0.15:
            machine_info = {
                'machine_type': row['Machine Type'],
                'machine_name': row['MACHINE'],
                'problem': row['Problem Description']
            }

            result_text = f"Root Cause: {row['Root Cause']}. Action Taken: {row['Action Taken']}"
            results.append((result_text, final_score, machine_info))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]

def vector_search(df, problem, machine_name, top_n=5, threshold=0.5):
    """Search using vector similarity - UNCHANGED"""
    query_text = f"{machine_name} {problem}"
    query_embedding = embedding_model.encode([query_text])[0]

    results = []

    if df is reference_df:
        similarities = cosine_similarity([query_embedding], reference_embeddings)[0]

        for i, score in enumerate(similarities):
            if score >= threshold:
                row = df.iloc[i]
                machine_info = {
                    'machine_type': row['Machine Type'],
                    'machine_name': row['MACHINE'],
                    'problem': row['Problem Description']
                }
                result_text = f"Root Cause: {row['Root Cause']}. Action Taken: {row['Action Taken']}"
                results.append((result_text, float(score), machine_info))
    else:
        for _, row in df.iterrows():
            row_text = f"{row['Machine Type']} {row['MACHINE']} {row['Problem Description']}"
            row_embedding = embedding_model.encode([row_text])[0]
            similarity = cosine_similarity([query_embedding], [row_embedding])[0][0]

            if similarity >= threshold:
                machine_info = {
                    'machine_type': row['Machine Type'],
                    'machine_name': row['MACHINE'],
                    'problem': row['Problem Description']
                }
                result_text = f"Root Cause: {row['Root Cause']}. Action Taken: {row['Action Taken']}"
                results.append((result_text, float(similarity), machine_info))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]

def generate_ai_response(machine_type, machine_name, problem_description, context_results=None):
    """
    Generate response using Mixtral model with context from semantic search
    REPLACES old Flan-T5 generation

    Args:
        machine_type: Type of machine
        machine_name: Specific machine name
        problem_description: Problem description
        context_results: Similar cases from semantic search (optional)

    Returns:
        Structured troubleshooting response
    """

    if not MIXTRAL_AVAILABLE:
        return _fallback_response(machine_type, machine_name, problem_description)

    try:
        # Get Mixtral model instance
        model = get_model()

        # Prepare context examples if available
        context_examples = []
        if context_results:
            for result_text, score, machine_info in context_results:
                # Parse result_text to extract root cause and action
                parts = result_text.split('Action Taken:')
                root_cause = parts[0].replace('Root Cause:', '').strip() if len(parts) > 0 else 'Unknown'
                action = parts[1].strip() if len(parts) > 1 else 'Unknown'

                context_examples.append({
                    'machine_name': machine_info.get('machine_name', 'Unknown'),
                    'problem': machine_info.get('problem', 'Unknown'),
                    'root_cause': root_cause,
                    'action': action,
                    'confidence': score
                })

        # Generate response
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
    """Fallback response when Mixtral is unavailable"""
    return (
        f"Root Cause: The {problem_description} in {machine_name} ({machine_type}) "
        f"typically indicates a malfunction in either the control system or mechanical components. "
        f"Possible Solutions: 1) Power cycle the machine completely. "
        f"2) Check for error codes on the control panel. "
        f"3) Inspect mechanical components related to the {problem_description}. "
        f"4) Verify all connections are secure. "
        f"5) Consult technical manual section for {machine_type} troubleshooting."
    )

def enhance_problem_description(problem, machine_type, machine_name):
    """Enhance the problem description for better matching - UNCHANGED"""
    enhanced = problem.strip()

    if machine_name.lower() not in enhanced.lower():
        enhanced = f"{machine_name} {enhanced}"

    common_variants = {
        "not working": ["doesn't work", "isn't working", "non-functional", "not functioning"],
        "alarm": ["alert", "warning", "beep", "notification"],
        "display": ["screen", "monitor", "lcd", "led screen"],
        "button": ["key", "switch", "control", "keypad"],
        "error": ["fault", "failure", "issue", "problem", "err", "trouble"],
    }

    for standard, variants in common_variants.items():
        for variant in variants:
            if variant in enhanced.lower() and standard not in enhanced.lower():
                enhanced = f"{enhanced} {standard}"
                break

    return enhanced

print("✓ Machine troubleshooting system loaded successfully!")
