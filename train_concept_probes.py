#Train linear probes to detect concepts in model activations using sklearn.
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error
import pandas as pd


def load_data(activations_path: str, meta_path: str):

    activations = np.load(activations_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    assert len(activations) == len(metadata), \
        f"Mismatch: {len(activations)} activations vs {len(metadata)} metadata entries"
    
    return activations, metadata


def extract_concept_labels(metadata: list[dict], concept: str):
    
    labels = np.array([m[concept] for m in metadata])
    return labels


def train_binary_probe(X_train, y_train, X_test, y_test, max_iter=1000):
    """Train a logistic regression probe for binary classification.
    
    Returns
    -------
    results : dict
        Dictionary with accuracy, f1_score, and the trained model.
    """
    # Handle class imbalance
    class_counts = np.bincount(y_train.astype(int))
    if len(class_counts) < 2 or min(class_counts) == 0:
        # Only one class present, return dummy results
        return {
            'accuracy': float(np.mean(y_test == y_train[0])),
            'f1_score': 0.0,
            'model': None,
            'warning': 'Only one class present in training data'
        }
    
    model = LogisticRegression(max_iter=max_iter, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'model': model,
    }


def train_regression_probe(X_train, y_train, X_test, y_test):
    """Train a ridge regression probe for continuous values.
    
    Returns
    -------
    results : dict
        Dictionary with r2_score, mae, and the trained model.
    """
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return {
        'r2_score': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'model': model,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activations-dir",
        type=str,
        default="activations",
        help="Directory containing activation .npy files",
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default="eval_meta.json",
        help="Path to metadata JSON file",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "ppo_door_key_subgoal_final",
            "ppo_door_key_subgoal_decay_final",
            "ppo_door_key_exploration_final",
        ],
        help="Model names (without _policy_activations.npy suffix)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="probe_results.csv",
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing",
    )
    args = parser.parse_args()
    
    # Define concepts to probe
    binary_concepts = [
        'has_key',
        'door_open',
        'door_locked',
        'key_visible',
        'door_visible',
        'goal_visible',
        'visited',
    ]
    
    continuous_concepts = [
        'dist_to_goal',
    ]
    
    # Load metadata once (shared across all models)
    print(f"Loading metadata from {args.meta_path}...")
    with open(args.meta_path, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} metadata entries")
    
    # Store results
    results = []
    
    # Process each model
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        # Load activations
        activations_path = Path(args.activations_dir) / f"{model_name}_policy_activations.npy"
        if not activations_path.exists():
            print(f"Warning: {activations_path} not found, skipping...")
            continue
        
        activations = np.load(activations_path)
        print(f"Loaded activations with shape {activations.shape}")
        
        # Train probes for binary concepts
        print("\nTraining binary concept probes:")
        for concept in binary_concepts:
            print(f"  - {concept}...", end=" ")
            
            # Extract labels
            labels = extract_concept_labels(metadata, concept)
            
            # Check if concept has variation
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                print(f"SKIP (only {len(unique_labels)} unique value(s))")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                activations, labels, test_size=args.test_size, random_state=42, stratify=labels
            )
            
            # Train probe
            probe_results = train_binary_probe(X_train, y_train, X_test, y_test)
            
            # Store results
            results.append({
                'model': model_name,
                'concept': concept,
                'concept_type': 'binary',
                'accuracy': probe_results['accuracy'],
                'f1_score': probe_results['f1_score'],
                'r2_score': None,
                'mae': None,
            })
            
            print(f"Acc={probe_results['accuracy']:.3f}, F1={probe_results['f1_score']:.3f}")
        
        # Train probes for continuous concepts
        print("\nTraining continuous concept probes:")
        for concept in continuous_concepts:
            print(f"  - {concept}...", end=" ")
            
            # Extract labels
            labels = extract_concept_labels(metadata, concept)
            
            # Filter out None values
            valid_mask = np.array([x is not None for x in labels])
            if not valid_mask.any():
                print("SKIP (all None values)")
                continue
            
            valid_activations = activations[valid_mask]
            valid_labels = labels[valid_mask].astype(float)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                valid_activations, valid_labels, test_size=args.test_size, random_state=42
            )
            
            # Train probe
            probe_results = train_regression_probe(X_train, y_train, X_test, y_test)
            
            # Store results
            results.append({
                'model': model_name,
                'concept': concept,
                'concept_type': 'continuous',
                'accuracy': None,
                'f1_score': None,
                'r2_score': probe_results['r2_score'],
                'mae': probe_results['mae'],
            })
            
            print(f"R²={probe_results['r2_score']:.3f}, MAE={probe_results['mae']:.3f}")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\n{'='*60}")
    print(f"✓ Results saved to {args.output}")
    print(f"{'='*60}")
    
    # Print summary comparison
    print("\n" + "="*60)
    print("SUMMARY: Binary Concept Accuracy by Model")
    print("="*60)
    binary_df = df[df['concept_type'] == 'binary']
    if not binary_df.empty:
        pivot = binary_df.pivot(index='concept', columns='model', values='accuracy')
        print(pivot.to_string())
    
    print("\n" + "="*60)
    print("SUMMARY: Continuous Concept R² by Model")
    print("="*60)
    continuous_df = df[df['concept_type'] == 'continuous']
    if not continuous_df.empty:
        pivot = continuous_df.pivot(index='concept', columns='model', values='r2_score')
        print(pivot.to_string())


if __name__ == "__main__":
    main()
