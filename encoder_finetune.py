import einops
import os
import random
from collections import deque
from pathlib import Path
import datasets
import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
import wandb
from utils.video import VideoRecorder
from datetime import timedelta
from torch.utils.data import DataLoader
import pickle
from datasets.core import TrajectoryEmbeddingDataset, split_traj_datasets
from datasets.vqbet_repro import TrajectorySlicerDataset
from accelerate.logging import get_logger
from accelerate import InitProcessGroupKwargs, DistributedDataParallelKwargs
from models.policy_mlp import policy_mlp
import math
import random
import torch.optim as optim
from torch.utils.data import Subset
import torch.nn as nn

import csv
import warnings
import glob






if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)

class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg
        
        process_group_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(seconds=cfg.train.timeout_seconds)
        )
        
        dist_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        
        self.accelerator = Accelerator(
                    log_with="wandb", kwargs_handlers=[process_group_kwargs, dist_kwargs]
                )
    
    def _split_and_slice_dataset(self, dataset):
        kwargs = {
            "train_fraction": self.cfg.train.train_fraction,
            "random_seed": self.cfg.seed,
            "window_size": self.cfg.train.window_size,
            "future_conditional": (self.cfg.train.goal_conditional == "future"),
            "min_future_sep": self.cfg.train.min_future_sep,
            "future_seq_len": self.cfg.train.goal_seq_len,
            "num_extra_predicted_actions": self.cfg.train.num_extra_predicted_actions,
        }
        return datasets.core.get_train_val_sliced(dataset, **kwargs)
    
    def _setup_loaders(self, batch_size=None, pin_memory=True, num_workers=None):
        print("BATCH SIZE", batch_size)
        if num_workers is None:
            num_workers = self.cfg.train.num_workers
        kwargs = {
            "batch_size": batch_size or self.cfg.train.batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        # scale batch size by number of gpus
        assert kwargs["batch_size"] % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Got {kwargs['batch_size']} and {self.accelerator.num_processes}."
        )
        kwargs["batch_size"] = kwargs["batch_size"] // self.accelerator.num_processes
        self.train_loader = DataLoader(self.train_set, shuffle=True, **kwargs)
        self.test_loader = DataLoader(self.test_set, shuffle=False, **kwargs)

        self.train_loader = self.accelerator.prepare(self.train_loader)
        self.test_loader = self.accelerator.prepare(self.test_loader)
    
    def setup_mrl(self):
        self.mrl_max_size = int(math.log2(self.cfg.eval_encoder.output_dim))
        self.mrl_min_size = self.cfg.mrl.min_size
        self.mrl_policies = []
        
        parameters = [{'params': self.encoder.parameters(), 'lr': 1e-4}]

        
        for power in range(self.mrl_min_size, self.mrl_max_size + 1):
            # print(power)

            self.cfg.policy_mlp.obs_dim = 2 ** power
            policy = hydra.utils.instantiate(self.cfg.policy_mlp).to(self.cfg.device)
            
            parameters.append({'params': policy.parameters(), 'lr': 1e-3})
            
            self.mrl_policies.append(policy)
            
        self.mrl_sizes = len(self.mrl_policies)
        self.mrl_optimizer = optim.Adam(parameters)

            
    def train(self, load = False):
        
        if load:
            encoder_path = '/home/harsh/arjun/dynamo_ssl/mrl_enc_weights/mrl_enc_weights_v1.pt'
            state_dict = torch.load(encoder_path)
            self.encoder.load_state_dict(state_dict)
            self.encoder = self.encoder.to(self.cfg.device)
            
            return
        
        
        self.criterion = nn.MSELoss()
        
        self.encoder.train()
        for policy in self.mrl_policies:
            policy.train()  
        
        optimizer = self.mrl_optimizer

        for epoch in tqdm.trange(self.cfg.train.epochs):
            
            # indices = np.random.randint(0, self.mrl_sizes, size=(len(self.train_loader)))
            # print(len(self.train_loader))
            batch = 0
            
            for data in tqdm.tqdm(self.train_loader):
                
                
                
                optimizer.zero_grad()
                
                obs, act, _ = (x.to(self.cfg.device) for x in data)

                encoded_obs = self.encoder(obs)

                obs = einops.rearrange(encoded_obs, "N T V E -> N T (V E)")
                
                
                total_loss = 0
                for policy in self.mrl_policies:
                    predicted_act = policy(obs)
                    total_loss += self.criterion(predicted_act, act)
                
                total_loss.backward()
                optimizer.step()  
                
                
                log_data([batch, total_loss.item()], 'train')
                
                batch += 1
            
    def evaluate(self):
  
        self.encoder.eval()  
        for policy in self.mrl_policies:
            policy.eval()  

        evaluation_losses = []  
        criterion = nn.MSELoss() 

        with torch.no_grad():  
            for size, policy_head in enumerate(self.mrl_policies):
                total_loss = 0.0
                num_batches = 0
                
                width = (self.mrl_min_size ** 2) * (2 ** size)

                for data in tqdm.tqdm(self.test_loader, desc=f"Evaluating policy {width}"):
                    obs, act, _ = (x.to(self.cfg.device) for x in data)

                    encoded_obs = self.encoder(obs)
                    obs = einops.rearrange(encoded_obs, "N T V E -> N T (V E)")
                    predicted_act = policy_head(obs)

                    loss = criterion(predicted_act, act)
                    total_loss += loss.item()
                    num_batches += 1

                average_loss = total_loss / num_batches
                evaluation_losses.append(average_loss)
                print(f"Policy {width}: Average Evaluation Loss = {average_loss}")

        for size, loss in enumerate(evaluation_losses):
            log_data([size, loss], 'eval')


        return evaluation_losses
    
    @torch.no_grad()
    def eval_on_env(self, policy):
        epoch = None
        num_evals = self.cfg.eval.num_env_evals
        num_eval_per_goal = 1
        videorecorder = None
        
        
        def goal_fn(goal_idx):
            if "use_libero_goal" in self.cfg.data:
                return goals_cache[goal_idx]
            else:
                return torch.zeros(1)
        
        self.env = hydra.utils.instantiate(self.cfg.env.gym)
        
        def embed(enc, obs):
            obs = (
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.cfg.device)
            )  # 1 V C H W
            result = enc(obs)
            result = einops.rearrange(result, "1 V E -> (V E)")
            return result

        avg_reward = 0
        action_list = []
        completion_id_list = []
        avg_max_coverage = []
        avg_final_coverage = []
        self.env.seed(self.cfg.seed)
        for goal_idx in range(num_evals):
            if videorecorder is not None:
                videorecorder.init(enabled=True)
            for i in range(num_eval_per_goal):
                obs_stack = deque(maxlen=self.cfg.eval.eval_window_size)
                this_obs = self.env.reset(goal_idx=goal_idx)  # V C H W
                assert (
                    this_obs.min() >= 0 and this_obs.max() <= 1
                ), "expect 0-1 range observation"
                this_obs_enc = embed(self.encoder, this_obs)
                obs_stack.append(this_obs_enc)
                done, step, total_reward = False, 0, 0
                goal = goal_fn(goal_idx)  # V C H W
                while not done:
                    obs = torch.stack(tuple(obs_stack)).float().to(self.cfg.device)
                    goal = torch.as_tensor(goal, dtype=torch.float32, device=self.cfg.device)
                    # goal = embed(encoder, goal)
                    goal = goal.unsqueeze(0).repeat(self.cfg.eval.eval_window_size, 1)
                    action = policy(obs.unsqueeze(0))
                    # print('action shape:', action.shape)
                    action = action[0]  # remove batch dim; always 1
                    # print('trim action shape:', action.shape)
                    
                    if self.cfg.action_window_size > 1:
                        action_list.append(action[-1].cpu().detach().numpy())
                        if len(action_list) > self.cfg.action_window_size:
                            action_list = action_list[1:]
                        curr_action = np.array(action_list)
                        curr_action = (
                            np.sum(curr_action, axis=0)[0] / curr_action.shape[0]
                        )
                        new_action_list = []
                        for a_chunk in action_list:
                            new_action_list.append(
                                np.concatenate(
                                    (a_chunk[1:], np.zeros((1)))
                                )
                            )
                        action_list = new_action_list
                    else:
                        curr_action = action[-1, 0, :].cpu().detach().numpy()

                    this_obs, reward, done, info = self.env.step(curr_action)
                    this_obs_enc = embed(self.encoder, this_obs)
                    obs_stack.append(this_obs_enc)

                    if videorecorder and videorecorder.enabled:
                        videorecorder.record(info["image"])
                    step += 1
                    total_reward += reward
                    goal = goal_fn(goal_idx)
                avg_reward += total_reward
                if self.cfg.env.gym.id == "pusht":
                    self.env.env._seed += 1
                    avg_max_coverage.append(info["max_coverage"])
                    avg_final_coverage.append(info["final_coverage"])
                elif self.cfg.env.gym.id == "blockpush":
                    avg_max_coverage.append(info["moved"])
                    avg_final_coverage.append(info["entered"])
                completion_id_list.append(info["all_completions_ids"])
            if videorecorder:
                videorecorder.save("eval_{}_{}.mp4".format(epoch, goal_idx))
        return (
            avg_reward / (num_evals * num_eval_per_goal),
            completion_id_list,
            avg_max_coverage,
            avg_final_coverage,
        )
    
    def save_encoder_weights(self, base_filename="mrl_enc_weights"):
        
        save_dir = "/home/harsh/arjun/dynamo_ssl/mrl_enc_weights"
        
        os.makedirs(save_dir, exist_ok=True)

        existing_files = [f for f in os.listdir(save_dir) if f.startswith(base_filename) and f.endswith(".pt")]

        version_numbers = []
        for file in existing_files:
            try:
                version = int(file.split("_v")[-1].split(".pt")[0])
                version_numbers.append(version)
            except (IndexError, ValueError):
                continue

        next_version = max(version_numbers, default=0) + 1

        save_filename = f"{base_filename}_v{next_version}.pt"
        save_path = os.path.join(save_dir, save_filename)

        # Save the encoder weights
        torch.save(self.encoder.state_dict(), save_path)
        print(f"Encoder weights saved to: {save_path}")


    def run(self):
        print(OmegaConf.to_yaml(self.cfg))
        seed_everything(self.cfg.seed)

        self.encoder = hydra.utils.instantiate(self.cfg.eval_encoder)
        
        encoder_path = '/home/harsh/arjun/dynamo_ssl/exp_local/2024.11.10/221233_train_pusht_dynamo/encoder.pt'
        self.encoder = torch.load(encoder_path).to(self.cfg.device)


        dataset = hydra.utils.instantiate(self.cfg.dataset)
        
            

        self.dataset = hydra.utils.instantiate(self.cfg.env.dataset)
        self.train_set, self.test_set = self._split_and_slice_dataset(self.dataset)
        
        subset_frac = self.cfg.subset_fraction
        
        # indices = list(range(int(len(self.train_set) * subset_frac)))
        # self.train_set = Subset(self.train_set, indices)
        # indices = list(range(int(len(self.test_set) * subset_frac)))
        # self.test_set = Subset(self.test_set, indices)

        self._setup_loaders(batch_size=self.cfg.train.batch_size)
        
        
    
        for param in self.encoder.parameters():
            param.requires_grad = True
            
        self.encoder.train()
        

        self.setup_mrl()
        
        
        # while True:
        #     pass

        # for idx, policy in enumerate(self.mrl_policies):
        #     size = (2 ** self.mrl_min_size) * (2 ** idx)
            
        #     online_data = self.eval_on_env(policy)
        #     print(online_data)
            
        #     log_data(list(online_data), f'online policy eval: {size}')
        
        self.evaluate()

        for idx, policy in enumerate(self.mrl_policies):
            size = (2 ** self.mrl_min_size) * (2 ** idx)
            
            online_data = self.eval_on_env(policy)
            print(online_data)
            
            log_data(list(online_data), f'online policy eval: {size}')
            
        self.train()
        
        self.evaluate()
        
        policy_data = []
        
        for idx, policy in enumerate(self.mrl_policies):
            size = (2 ** self.mrl_min_size) * (2 ** idx)
            
            online_data = self.eval_on_env(policy)
            print(online_data)
            
            log_data(list(online_data), f'online policy eval: {size}')
            
        self.save_encoder_weights()
            
            
            
        
            

        

        # cbet_model = hydra.utils.instantiate(self.cfg.model).to(self.cfg.device)
        
        # optimizer = cbet_model.configure_optimizers(
        #     weight_decay=cfg.optim.weight_decay,
        #     learning_rate=cfg.optim.lr,
        #     betas=cfg.optim.betas,
        # )
        
        # env = hydra.utils.instantiate(cfg.env.gym)
        
        # if "use_libero_goal" in cfg.data:
        #     with torch.no_grad():
        #         # calculate goal embeddings for each task
        #         goals_cache = []
        #         for i in range(10):
        #             idx = i * 50
        #             last_obs, _, _ = dataset.get_frames(idx, [-1])  # 1 V C H W
        #             last_obs = last_obs.to(cfg.device)
        #             embd = encoder(last_obs)[0]  # V E
        #             embd = einops.rearrange(embd, "V E -> (V E)")
        #             goals_cache.append(embd)

        #     def goal_fn(goal_idx):
        #         return goals_cache[goal_idx]
        # else:
        #     empty_tensor = torch.zeros(1)

        #     def goal_fn(goal_idx):
        #         return empty_tensor
            
            

        # run = wandb.init(
        #     project=cfg.wandb.project,
        #     entity=cfg.wandb.entity,
        #     config=OmegaConf.to_container(cfg, resolve=True),
        # )
        
        
        # run_name = run.name or "Offline"
        # save_path = Path(cfg.save_path) / run_name
        # save_path.mkdir(parents=True, exist_ok=False)
        # video = VideoRecorder(dir_name=save_path)

        # @torch.no_grad()
        # def eval_on_env(
        #     cfg,
        #     num_evals=cfg.num_env_evals,
        #     num_eval_per_goal=1,
        #     videorecorder=None,
        #     epoch=None,
        # ):
        #     def embed(enc, obs):
        #         obs = (
        #             torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(cfg.device)
        #         )  # 1 V C H W
        #         result = enc(obs)
        #         result = einops.rearrange(result, "1 V E -> (V E)")
        #         return result

        #     avg_reward = 0
        #     action_list = []
        #     completion_id_list = []
        #     avg_max_coverage = []
        #     avg_final_coverage = []
        #     env.seed(cfg.seed)
        #     for goal_idx in range(num_evals):
        #         if videorecorder is not None:
        #             videorecorder.init(enabled=True)
        #         for i in range(num_eval_per_goal):
        #             obs_stack = deque(maxlen=cfg.eval_window_size)
        #             this_obs = env.reset(goal_idx=goal_idx)  # V C H W
        #             assert (
        #                 this_obs.min() >= 0 and this_obs.max() <= 1
        #             ), "expect 0-1 range observation"
        #             this_obs_enc = embed(encoder, this_obs)
        #             obs_stack.append(this_obs_enc)
        #             done, step, total_reward = False, 0, 0
        #             goal = goal_fn(goal_idx)  # V C H W
        #             while not done:
        #                 obs = torch.stack(tuple(obs_stack)).float().to(cfg.device)
        #                 goal = torch.as_tensor(goal, dtype=torch.float32, device=cfg.device)
        #                 # goal = embed(encoder, goal)
        #                 goal = goal.unsqueeze(0).repeat(cfg.eval_window_size, 1)
        #                 action, _, _ = cbet_model(obs.unsqueeze(0), goal.unsqueeze(0), None)
        #                 action = action[0]  # remove batch dim; always 1
        #                 if cfg.action_window_size > 1:
        #                     action_list.append(action[-1].cpu().detach().numpy())
        #                     if len(action_list) > cfg.action_window_size:
        #                         action_list = action_list[1:]
        #                     curr_action = np.array(action_list)
        #                     curr_action = (
        #                         np.sum(curr_action, axis=0)[0] / curr_action.shape[0]
        #                     )
        #                     new_action_list = []
        #                     for a_chunk in action_list:
        #                         new_action_list.append(
        #                             np.concatenate(
        #                                 (a_chunk[1:], np.zeros((1, a_chunk.shape[1])))
        #                             )
        #                         )
        #                     action_list = new_action_list
        #                 else:
        #                     curr_action = action[-1, 0, :].cpu().detach().numpy()

        #                 this_obs, reward, done, info = env.step(curr_action)
        #                 this_obs_enc = embed(encoder, this_obs)
        #                 obs_stack.append(this_obs_enc)

        #                 if videorecorder.enabled:
        #                     videorecorder.record(info["image"])
        #                 step += 1
        #                 total_reward += reward
        #                 goal = goal_fn(goal_idx)
        #             avg_reward += total_reward
        #             if cfg.env.gym.id == "pusht":
        #                 env.env._seed += 1
        #                 avg_max_coverage.append(info["max_coverage"])
        #                 avg_final_coverage.append(info["final_coverage"])
        #             elif cfg.env.gym.id == "blockpush":
        #                 avg_max_coverage.append(info["moved"])
        #                 avg_final_coverage.append(info["entered"])
        #             completion_id_list.append(info["all_completions_ids"])
        #         videorecorder.save("eval_{}_{}.mp4".format(epoch, goal_idx))
        #     return (
        #         avg_reward / (num_evals * num_eval_per_goal),
        #         completion_id_list,
        #         avg_max_coverage,
        #         avg_final_coverage,
        #     )

        # metrics_history = []
        # reward_history = []
        # for epoch in tqdm.trange(cfg.epochs):
        #     cbet_model.eval()
        #     if epoch % cfg.eval_on_env_freq == 0:
        #         avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
        #             cfg,
        #             videorecorder=video,
        #             epoch=epoch,
        #             num_eval_per_goal=cfg.num_final_eval_per_goal,
        #         )
        #         reward_history.append(avg_reward)
        #         with open("{}/completion_idx_{}.json".format(save_path, epoch), "wb") as fp:
        #             pickle.dump(completion_id_list, fp)
        #         wandb.log({"eval_on_env": avg_reward})
        #         if cfg.env.gym.id in ["pusht", "blockpush"]:
        #             metric_final = (
        #                 "final coverage" if cfg.env.gym.id == "pusht" else "entered"
        #             )
        #             metric_max = "max coverage" if cfg.env.gym.id == "pusht" else "moved"
        #             metrics = {
        #                 f"{metric_final} mean": sum(final_coverage) / len(final_coverage),
        #                 f"{metric_final} max": max(final_coverage),
        #                 f"{metric_final} min": min(final_coverage),
        #                 f"{metric_max} mean": sum(max_coverage) / len(max_coverage),
        #                 f"{metric_max} max": max(max_coverage),
        #                 f"{metric_max} min": min(max_coverage),
        #             }
        #             wandb.log(metrics)
        #             metrics_history.append(metrics)

        #     if epoch % cfg.eval_freq == 0:
        #         total_loss = 0
        #         action_diff = 0
        #         action_diff_tot = 0
        #         action_diff_mean_res1 = 0
        #         action_diff_mean_res2 = 0
        #         action_diff_max = 0
        #         with torch.no_grad():
        #             for data in test_loader:
        #                 obs, act, goal = (x.to(cfg.device) for x in data)
        #                 assert obs.ndim == 4, "expect N T V E here"
        #                 obs = einops.rearrange(obs, "N T V E -> N T (V E)")
        #                 goal = einops.rearrange(goal, "N T V E -> N T (V E)")
        #                 predicted_act, loss, loss_dict = cbet_model(obs, goal, act)
        #                 total_loss += loss.item()
        #                 wandb.log({"eval/{}".format(x): y for (x, y) in loss_dict.items()})
        #                 action_diff += loss_dict["action_diff"]
        #                 action_diff_tot += loss_dict["action_diff_tot"]
        #                 action_diff_mean_res1 += loss_dict["action_diff_mean_res1"]
        #                 action_diff_mean_res2 += loss_dict["action_diff_mean_res2"]
        #                 action_diff_max += loss_dict["action_diff_max"]
        #         print(f"Test loss: {total_loss / len(test_loader)}")
        #         wandb.log({"eval/epoch_wise_action_diff": action_diff})
        #         wandb.log({"eval/epoch_wise_action_diff_tot": action_diff_tot})
        #         wandb.log({"eval/epoch_wise_action_diff_mean_res1": action_diff_mean_res1})
        #         wandb.log({"eval/epoch_wise_action_diff_mean_res2": action_diff_mean_res2})
        #         wandb.log({"eval/epoch_wise_action_diff_max": action_diff_max})

        #     cbet_model.train()
        #     for data in tqdm.tqdm(train_loader):
        #         optimizer.zero_grad()
        #         obs, act, goal = (x.to(cfg.device) for x in data)
        #         obs = einops.rearrange(obs, "N T V E -> N T (V E)")
        #         goal = einops.rearrange(goal, "N T V E -> N T (V E)")
        #         predicted_act, loss, loss_dict = cbet_model(obs, goal, act)
        #         wandb.log({"train/{}".format(x): y for (x, y) in loss_dict.items()})
        #         loss.backward()
        #         optimizer.step()

        # avg_reward, completion_id_list, max_coverage, final_coverage = eval_on_env(
        #     cfg,
        #     num_evals=cfg.num_final_evals,
        #     num_eval_per_goal=cfg.num_final_eval_per_goal,
        #     videorecorder=video,
        #     epoch=cfg.epochs,
        # )
        # reward_history.append(avg_reward)
        # if cfg.env.gym.id in ["pusht", "blockpush"]:
        #     metric_final = "final coverage" if cfg.env.gym.id == "pusht" else "entered"
        #     metric_max = "max coverage" if cfg.env.gym.id == "pusht" else "moved"
        #     metrics = {
        #         f"{metric_final} mean": sum(final_coverage) / len(final_coverage),
        #         f"{metric_final} max": max(final_coverage),
        #         f"{metric_final} min": min(final_coverage),
        #         f"{metric_max} mean": sum(max_coverage) / len(max_coverage),
        #         f"{metric_max} max": max(max_coverage),
        #         f"{metric_max} min": min(max_coverage),
        #     }
        #     wandb.log(metrics)
        #     metrics_history.append(metrics)

        # with open("{}/completion_idx_final.json".format(save_path), "wb") as fp:
        #     pickle.dump(completion_id_list, fp)
        # if cfg.env.gym.id == "pusht":
        #     final_eval_on_env = max([x["final coverage mean"] for x in metrics_history])
        # elif cfg.env.gym.id == "blockpush":
        #     final_eval_on_env = max([x["entered mean"] for x in metrics_history])
        # elif cfg.env.gym.id == "libero_goal":
        #     final_eval_on_env = max(reward_history)
        # elif cfg.env.gym.id == "kitchen-v0":
        #     final_eval_on_env = avg_reward
        # wandb.log({"final_eval_on_env": final_eval_on_env})
        # return final_eval_on_env


def log_data(data, str = ''):
    log_file_path = '/home/harsh/arjun/dynamo_ssl/log_files/test.txt'
    with open(log_file_path, "a") as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow([str] + data)


@hydra.main(config_path="configs", version_base="1.2", config_name = 'pusht_finetune')
def main(cfg):
    # warnings.filterwarnings("ignore", category=FutureWarning)

    trainer = Trainer(cfg)
    trainer.run()
    
    
if __name__ == "__main__":
    main()