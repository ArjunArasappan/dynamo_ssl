import einops
import os
import random
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import tqdm
from omegaconf import OmegaConf

import wandb
from utils.video import VideoRecorder
import pickle
from datasets.core import TrajectoryEmbeddingDataset, split_traj_datasets
from datasets.vqbet_repro import TrajectorySlicerDataset

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


@hydra.main(config_path="finetune_configs", version_base="1.2")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    # Initialize and prepare the encoder for training
    encoder = hydra.utils.instantiate(cfg.encoder)
    encoder = encoder.to(cfg.device).train()  # Set encoder to train mode

    dataset = hydra.utils.instantiate(cfg.dataset)
    train_data, test_data = split_traj_datasets(
        dataset,
        train_fraction=cfg.train_fraction,
        random_seed=cfg.seed,
    )
    use_libero_goal = cfg.data.get("use_libero_goal", False)
    train_data = TrajectoryEmbeddingDataset(
        encoder, train_data, device=cfg.device, embed_goal=use_libero_goal
    )
    test_data = TrajectoryEmbeddingDataset(
        encoder, test_data, device=cfg.device, embed_goal=use_libero_goal
    )
    traj_slicer_kwargs = {
        "window": cfg.data.window_size,
        "action_window": cfg.data.action_window_size,
        "vqbet_get_future_action_chunk": cfg.data.vqbet_get_future_action_chunk,
        "future_conditional": (cfg.data.goal_conditional == "future"),
        "min_future_sep": cfg.data.action_window_size,
        "future_seq_len": cfg.data.future_seq_len,
        "use_libero_goal": use_libero_goal,
    }
    train_data = TrajectorySlicerDataset(train_data, **traj_slicer_kwargs)
    test_data = TrajectorySlicerDataset(test_data, **traj_slicer_kwargs)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False, pin_memory=False
    )

    input_dim = cfg.encoder.output_dim + cfg.goal_dim 
    
    min_mrl = cfg.mrl.min_emb
    
    mrl_sizes = []

    while min_mrl <= input_dim:
        mrl_sizes.append(min_mrl)
        min_mrl *= 2
        
    print(mrl_sizes)
        
    policy_heads = {}

    
    for size in mrl_sizes:
        print('policy', size)
        policy = hydra.utils.instantiate(cfg.policy_mlp).to(cfg.device)
        policy_heads[size] = policy
    

    optimizers = {}
    
    for size in mrl_sizes:
        policy = policy_heads[size]
        print('opt', size)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(policy.parameters()),  # fine-tune both
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
            betas=cfg.optim.betas,
        )
        
        optimizers[i] = optimizer
        
    print('done')

    env = hydra.utils.instantiate(cfg.env.gym)
    
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run_name = run.name or "FineTuneWithMLP"
    save_path = Path(cfg.save_path) / run_name
    save_path.mkdir(parents=True, exist_ok=False)
    video = VideoRecorder(dir_name=save_path)

    metrics_history = []
    reward_history = []
    
    print('done')

    # Training loop with fine-tuning
    for epoch in tqdm.trange(cfg.epochs):
        encoder.train()
        policy_model.train()

        for data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            obs, act, goal = (x.to(cfg.device) for x in data)
            obs = einops.rearrange(obs, "N T V E -> N T (V E)")
            goal = einops.rearrange(goal, "N T V E -> N T (V E)")
            
            # Forward pass through encoder and MLP
            encoded_obs = encoder(obs)
            mlp_input = torch.cat([encoded_obs, goal], dim=-1)
            predicted_act = policy_model(mlp_input)
            loss = nn.MSELoss()(predicted_act, act)  # Loss based on action prediction

            # Backpropagation for fine-tuning
            loss.backward()
            optimizer.step()
            wandb.log({"train/loss": loss.item()})

        # Validation and evaluation loop
        policy_model.eval()
        encoder.eval()
        if epoch % cfg.eval_freq == 0:
            total_loss = 0
            with torch.no_grad():
                for data in test_loader:
                    obs, act, goal = (x.to(cfg.device) for x in data)
                    obs = einops.rearrange(obs, "N T V E -> N T (V E)")
                    goal = einops.rearrange(goal, "N T V E -> N T (V E)")
                    encoded_obs = encoder(obs)
                    mlp_input = torch.cat([encoded_obs, goal], dim=-1)
                    predicted_act = policy_model(mlp_input)
                    loss = nn.MSELoss()(predicted_act, act)
                    total_loss += loss.item()
                wandb.log({"eval/loss": total_loss / len(test_loader)})
            print(f"Validation loss at epoch {epoch}: {total_loss / len(test_loader)}")

        # Evaluation in the environment (e.g., episodic reward tracking)
        if epoch % cfg.eval_on_env_freq == 0:
            avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
                cfg,
                videorecorder=video,
                epoch=epoch,
                num_eval_per_goal=cfg.num_final_eval_per_goal,
            )
            reward_history.append(avg_reward)
            wandb.log({"eval_on_env": avg_reward})

    # Save the final fine-tuned models
    torch.save(encoder.state_dict(), save_path / "fine_tuned_encoder.pth")
    torch.save(policy_model.state_dict(), save_path / "fine_tuned_policy_model.pth")

if __name__ == "__main__":
    main()
