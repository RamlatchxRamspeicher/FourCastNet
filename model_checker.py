import argparse
import os
import torch
from FourCastNet.utils.YParams import YParams
from networks.afnonet_mp_v1 import AFNONetDist
def check_model_parameters(checkpoint_path):
    # Load the model checkpoint
    model = AFNONetDist(
        params,
        img_size = (720, 1440),
        patch_size = (8, 8),
        in_chans = 20,
        out_chans = 20,
        embed_dim = 768,
        depth = 12,
        mlp_ratio = 4,
        drop_rate = 0,
        drop_path_rate = 0,
        num_blocks = 8,
        sparsity_threshold = 0.01,
        hard_thresholding_fraction = 1.
    )
    model.load_state_dict(torch.load(checkpoint_path))
    
    # Iterate through the named parameters
    for n, p in model.get_named_parameters():
        print(f"{n}: \t finite params: {p.isfinite()} \t grads: {p.grad.isfinite()}")

if __name__ == "__main__":
    # Path to the model checkpoint
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument("--ckpt", default='default', type=str)
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    checkpoint_path = os.path.abspath(args.ckpt)
    # Check the model parameters
    check_model_parameters(checkpoint_path,params)