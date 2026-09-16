import torch
import torch.nn as nn
from typing import Tuple
from .DDQNscheduler import GRUQNetwork, MLPQNetwork


def make_device(use_cpu: bool, gpu_id: int) -> torch.device:
    if (not use_cpu) and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")

def load_checkpoint_and_model(ckpt_path: str, device: torch.device) -> Tuple[dict, nn.Module]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "q_net" not in ckpt:
        raise KeyError("Checkpoint does not contain 'q_net'. Expected DQN checkpoint.")

    sd = ckpt["q_net"]
    if any(k.startswith("gru.") for k in sd.keys()):
        meta = ckpt.get("model_meta", {})
        if meta:
            obs_dim = int(meta.get("obs_dim", sd["gru.weight_ih"].shape[1]))
            hidden_size = int(meta.get("hidden_size", sd["gru.weight_hh"].shape[1]))
            n_actions = int(meta.get("n_actions", sd["q_head.weight"].shape[0]))
        else:
            obs_dim = int(sd["gru.weight_ih"].shape[1])
            hidden_size = int(sd["gru.weight_hh"].shape[1])
            n_actions = int(sd["q_head.weight"].shape[0])

        model = GRUQNetwork(obs_dim=obs_dim, hidden_size=hidden_size, n_actions=n_actions).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        print(f"[MODEL] GRU obs_dim={obs_dim} hidden={hidden_size} actions={n_actions}")
        return ckpt, model

    if all(k in sd for k in ["net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"]):
        obs_dim = int(sd["net.0.weight"].shape[1])
        h1 = int(sd["net.0.weight"].shape[0])
        h2 = int(sd["net.2.weight"].shape[0])
        n_actions = int(sd["net.4.weight"].shape[0])

        model = MLPQNetwork(obs_dim=obs_dim, h1=h1, h2=h2, n_actions=n_actions).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        print(f"[MODEL] MLP obs_dim={obs_dim} h1={h1} h2={h2} actions={n_actions}")
        return ckpt, model

    raise ValueError("Unsupported q_net architecture in checkpoint.")