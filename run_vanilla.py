import argparse
import os
from graphcut_utils import *
from run_generative_photomontage import inference_gpm

BASE_MODEL_PATH = "runwayml/stable-diffusion-v1-5"
NUM_INFERENCE_STEPS = 20

def run_vanilla_batch(
    model_type,
    shape,
    prompts,
    seeds):
  cond_img_file = "./data/{}-{}.png".format(shape, model_type)
  output_qkv_dir = os.path.join(shape, model_type) # Save QKV features to this folder.

  if len(prompts) == 1:
     prompts = prompts * len(seeds)
  
  for i, seed in enumerate(seeds):
    inference_gpm(model_type, 
                    os.path.join(VANILLA_DIR, shape), 
                    cond_img_file, 
                    prompts[i], 
                    seed,
                    1, 
                    inject_self=False,
                    qkv_save_folder=output_qkv_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Vanilla ControlNet and save QKV features.')

    parser.add_argument("--cond", type=str, required=True, default="data/applelogo-canny.png", help="Input condition image with the format {NAME}-{MODEL}.png")
    parser.add_argument("--prompts", type=str, nargs="+", default="A rock on grass", help="One or more prompts for each image")
    parser.add_argument("--seeds", type=int, nargs="+", default=0, help="List of seeds, one for each image")
    args = parser.parse_args()

    basename = os.path.basename(args.cond).split(".")[0]
    name, model = basename.split("-")
    run_vanilla_batch(model, name, args.prompts, args.seeds)