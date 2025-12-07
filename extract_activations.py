"""Extract activations from trained PPO models on the evaluation dataset."""
import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from gymnasium.wrappers import FilterObservation, FlattenObservation

from train_env import make_env


def extract_activations(model_path: str, obs_array: np.ndarray, device: str = "cpu", layer: str = "policy"):
    """Extract activations from a PPO model's hidden layers.
    
    Parameters
    ----------
    model_path : str
        Path to the saved PPO model (.zip file).
    obs_array : np.ndarray
        Array of observations with shape (N, obs_dim).
    device : str
        Device to run inference on.
    layer : str
        Which layer to extract from: "policy", "value", or "shared".
        
    Returns
    -------
    activations : np.ndarray
        Extracted activations with shape (N, hidden_dim).
    """
    # Load the model
    model = PPO.load(model_path, device=device)
    model.policy.eval()
    
    activations = []
    
    # Process in batches to avoid memory issues
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(obs_array), batch_size):
            batch = obs_array[i:i+batch_size]
            # Convert to tensor and move to device
            obs_tensor = torch.FloatTensor(batch).to(device)
            
            # Extract features through the feature extractor
            features = model.policy.features_extractor(obs_tensor)
            
            # Pass through the MLP extractor to get hidden representations
            if layer == "shared":
                # For policies with shared layers, get the shared latent
                latent = model.policy.mlp_extractor.forward_actor(features)
            elif layer == "policy":
                # Get the policy network's hidden representation
                latent = model.policy.mlp_extractor.forward_actor(features)
            elif layer == "value":
                # Get the value network's hidden representation
                latent = model.policy.mlp_extractor.forward_critic(features)
            else:
                raise ValueError(f"Unknown layer: {layer}")
            
            activations.append(latent.cpu().numpy())
    
    return np.vstack(activations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-obs",
        type=str,
        default="eval_obs.npy",
        help="Path to evaluation observations (.npy file)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "runs/subgoal_door_key/ppo_door_key_subgoal_final.zip",
            "runs/subgoal_decay_fresh4/ppo_door_key_subgoal_decay_final.zip",
            "runs/exploration_more4_entropy/ppo_door_key_exploration_final.zip",
        ],
        help="Paths to trained PPO models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="activations",
        help="Directory to save extracted activations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (cpu or cuda)",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="policy",
        choices=["policy", "value", "shared"],
        help="Which layer to extract activations from",
    )
    args = parser.parse_args()
    
    # Load evaluation observations
    print(f"Loading evaluation observations from {args.eval_obs}...")
    eval_obs = np.load(args.eval_obs)
    print(f"Loaded {len(eval_obs)} observations with shape {eval_obs.shape}")
    
    # The eval_obs contains raw image observations (7, 7, 3)
    # We need to flatten them to match the model's expected input
    # The model expects FilterObservation + FlattenObservation output
    
    # Create a dummy env to get the observation space structure
    dummy_env = make_env(env_name="door_key", reward_wrapper=None)
    obs_sample, _ = dummy_env.reset()
    expected_shape = obs_sample.shape
    print(f"Expected observation shape from wrapped env: {expected_shape}")
    
    # Flatten the eval observations to match
    # eval_obs is (N, 7, 7, 3) but we need to extract image + direction
    # For now, we'll flatten just the image part and add a dummy direction
    N = len(eval_obs)
    flattened_obs = np.zeros((N, expected_shape[0]))
    
    # The wrapped env filters to ["image", "direction"] then flattens
    # Image is (7, 7, 3) = 147 dims, direction is one-hot (4 dims) = 151 total
    for i in range(N):
        img = eval_obs[i].flatten()  # 147 dims
        # We don't have direction info in eval_obs, so use 0 (facing right)
        direction = np.array([1, 0, 0, 0])  # one-hot for direction 0
        flattened_obs[i] = np.concatenate([img, direction])
    
    print(f"Flattened observations to shape {flattened_obs.shape}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract activations for each model
    for model_path in args.models:
        print(f"\nProcessing model: {model_path}")
        
        # Extract model name for output file
        model_name = Path(model_path).stem  # e.g., "ppo_door_key_subgoal_final"
        
        # Extract activations
        activations = extract_activations(model_path, flattened_obs, device=args.device, layer=args.layer)
        
        # Save activations
        output_path = output_dir / f"{model_name}_{args.layer}_activations.npy"
        np.save(output_path, activations)
        print(f"Saved activations with shape {activations.shape} to {output_path}")
    
    print(f"\n✓ All activations extracted and saved to {output_dir}/")


if __name__ == "__main__":
    main()
