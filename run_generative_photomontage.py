from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import argparse
import os
from PIL import Image
import numpy as np
import datetime
import pytz
from gen_photomontage_utils import *

# Initialize GPM Attention Processors
def get_attn_processor_dict(
        model, 
        control_image, 
        num_inference_steps, 
        save_img_latents='none',
        inject_after_timestep=0, 
        layers='all',
        inject_self=True,
        q_guidance_img_dir=None,
        q_inject_mask=None):
    
    attention_processors = {}

    for key in model.attn_processors.keys():
        attention_processors[key] = GPMAttentionProcessor(
            key, 
            control_image, 
            num_inference_steps, 
            save_img_latents,
            inject_after_timestep,
            layers,
            inject_self,
            q_guidance_img_dir,
            q_inject_mask)
        
    return attention_processors

def generate_photomontage(model_type, shape, prompts, seeds):
  cond_img_file = "./data/{}-{}.png".format(shape, model_type)
  qkv_inject_dir = None
  mask_files = None

  if len(prompts) == 1:
    prompts = prompts * len(seeds)
  elif len(prompts) > len(seeds):
    print("There are more prompts than seeds provided.")
    prompts = prompts[:len(seeds)]
    print("List of prompts truncated to: {}".format(prompts))

  timestamp = datetime.datetime.now(pytz.timezone('America/New_York'))
  timestamp_str = timestamp.strftime("%m%d-%H%M%S")

  mask_files = get_image_stack_masks(shape, model_type, prompts, seeds, use_gc_masks=True)
  log_composite_info(shape, model_type, prompts, seeds, timestamp_str)

  qkv_inject_dir = [get_qkv_dir(shape, model_type, prompts[0], seeds[0])]
  for i, s in enumerate(seeds):
    if i > 0:
      fn = get_mask_file(shape, model_type, prompts[i], s, gc=True)
      if os.path.isfile(fn):
        qkv_inject_dir.append(get_qkv_dir(shape, model_type, prompts[i], s))
      else:
        raise ValueError("QKV features missing. Please run vanilla ControlNet first with run_vanilla.py or demo.py.")

  composite_metadata = [timestamp_str, ",".join(map(str, seeds))]

  return inference_gpm(model_type,
                      os.path.join(COMPOSITE_DIR, shape), 
                      cond_img_file, 
                      prompt=prompts[0],
                      initial_seed=seeds[0],
                      inject_self=True,  # Set to True to enable self-attention injection.
                      qkv_save_folder=None,
                      qkv_inject_folder=qkv_inject_dir,
                      injection_masks=mask_files,
                      composite_metadata=composite_metadata)

def inference_gpm(
        model_type, 
        output_dir, 
        cond_img_file, 
        prompt, 
        initial_seed,
        num_seeds=1,
        inject_self=False,  # Set to False to run vanilla ControlNet. Set to True to enable attention injection.
        inject_after_timestep=0,
        layers='all',
        qkv_save_folder=None,
        qkv_inject_folder=None,
        injection_masks=None,
        composite_metadata=None):
    
    controlnet_path = "lllyasviel/sd-controlnet-{}".format(model_type)
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16, use_safetensors=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True)             
    control_image = load_image(cond_img_file)

    unet_attention_processors = get_attn_processor_dict(
            pipe.unet, 
            control_image,
            NUM_INFERENCE_STEPS, 
            qkv_save_folder,
            inject_after_timestep,
            layers,
            inject_self,
            qkv_inject_folder,
            injection_masks)
    pipe.unet.set_attn_processor(unet_attention_processors)

    controlnet_attention_processors = get_attn_processor_dict(
            pipe.controlnet, 
            control_image,
            NUM_INFERENCE_STEPS, 
            None, 
            inject_after_timestep,
            layers,
            False,
            None,
            None)
    pipe.controlnet.set_attn_processor(controlnet_attention_processors)

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.enable_model_cpu_offload()

    # Create output directory
    prompt_subdir =  "_".join(prompt.split())
    output_subdir = os.path.join(output_dir, model_type, prompt_subdir)
    os.makedirs(output_subdir, exist_ok=True)
    Image.fromarray(np.array(control_image)).save(os.path.join(output_subdir, "cond_img.jpg"))
        
    # Run inference
    return_images = []
    for i in range(initial_seed, initial_seed+num_seeds):
        generator = [torch.Generator().manual_seed(i)]
        timestamp_flag = "{}-".format(composite_metadata[0]) if composite_metadata is not None else ""
        composite_flag = "-comp_{}".format(composite_metadata[1]) if composite_metadata is not None else ""
        output_file = os.path.join(output_subdir, "{}seed{}{}.png".format(timestamp_flag, i, composite_flag))

        for k in pipe.unet.attn_processors.keys():
            pipe.unet.attn_processors[k].set_prompt(prompt)
            pipe.unet.attn_processors[k].set_seed(i)
            pipe.unet.attn_processors[k].set_component('unet')

        for k in pipe.controlnet.attn_processors.keys():
            pipe.controlnet.attn_processors[k].set_prompt(prompt)
            pipe.controlnet.attn_processors[k].set_seed(i)
            pipe.controlnet.attn_processors[k].set_component('control')
            
        output = pipe(prompt, 
                    num_inference_steps=NUM_INFERENCE_STEPS, 
                    generator=generator, 
                    image=control_image, 
                    num_images_per_prompt=len(generator))
        images = output.images

        images[0].save(output_file)
        print("Saved output to {}".format(output_file))
        return_images.append(images[0])

    return [control_image], return_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Generative Photomontage.')

    parser.add_argument("--cond", type=str, required=True, default="data/applelogo-canny.png", help="Input condition image with the format {NAME}-{MODEL}.png")
    parser.add_argument("--prompts", type=str, nargs="+", default="A rock on grass", help="One prompt for all images, or one prompt per image")
    parser.add_argument("--seeds", type=int, nargs="+", default=0, help="List of seeds, one for each image")

    args = parser.parse_args()
    

    basename = os.path.basename(args.cond).split(".")[0]
    name, model = basename.split("-")

    # Checks number of prompts:
    if len(args.prompts) > 1:
       assert len(args.prompts) == len(args.seeds), "Number of prompts ({}) should be 1 or equal to the number of seeds ({})".format(len(args.prompts), len(args.seeds))

    # Checks if stroke masks are in correct folder.
    info_str = ""
    for i, seed in enumerate(args.seeds):
        stroke_mask_file = get_mask_file(name, model, args.prompts[i] if len(args.prompts) > 1 else args.prompts[0], seed)
        if not os.path.isfile(stroke_mask_file):
            print("\nERROR: Stroke mask is missing: {}.".format(stroke_mask_file))
            print("Please make sure you have a stroke mask for each image, saved in {}/NAME/MODEL/PROMPT/masks. Filenames should have the format \"mask_SEED.png\"".format(
               COMPOSITE_DIR))
            exit()
        info_str += "Prompt: {}, Seed={}\n".format(args.prompts[i] if len(args.prompts) > 1 else args.prompts[0], seed)

    print("Running graph cut on:\n{}".format(info_str))
    compute_graph_cut(name, model, args.prompts, args.seeds)
    print("Done.")

    print("Blending with graph-cut results...")
    generate_photomontage(model, name, args.prompts, args.seeds)
    print("Done.")