import torch
from torch.utils.data import Dataset
from .cache import load_cache

class RealMotionCacheDataset(Dataset):
    def __init__(self,path):
        self.payload=load_cache(path); self.samples=self.payload["samples"]
    def __len__(self): return len(self.samples)
    def __getitem__(self,i): return self.samples[i]

def collate_real_motion(batch):
    keys=("moving_history_latent","future_moving_latent","static_future_latent","kta_future_latent","generation_support")
    out={k:torch.stack([x[k] for x in batch]) for k in keys}
    optional=("trajectory","confident_static_mask","gt_moving_support","planning_support","sample_id")
    for k in optional:
        if all(k in x for x in batch):
            out[k]=torch.stack([x[k] for x in batch]) if torch.is_tensor(batch[0][k]) else [x[k] for x in batch]
    return out
