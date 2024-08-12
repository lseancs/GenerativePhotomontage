from graphcut_utils import *
import shutil

BASE_MODEL_PATH = "runwayml/stable-diffusion-v1-5"
NUM_INFERENCE_STEPS = 20

def get_mask_file(shape, model_type, prompt, seed, gc=False, mask_suffix=None):
  suffix = ""
  if gc:
    suffix += "_gc"
  if mask_suffix is not None and mask_suffix != ".":
    suffix += "_{}".format(mask_suffix)
  fn = os.path.join(COMPOSITE_DIR,
                       shape,
                       model_type,
                       prompt.replace(" ", "_"),
                       "masks",
                       "mask_{}{}.png".format(seed, suffix))
  dirname = os.path.dirname(fn)
  os.makedirs(dirname, exist_ok=True)
  return fn

def get_prompt_dir(input_dir, shape, model_type, prompt):
  dirname = os.path.join(input_dir,
                      shape,
                      model_type,
                      prompt.replace(" ", "_"))
  return dirname


# Get graph cut labels
def get_gc_file(shape, model_type, prompt, seeds, header="label", suffix='npy'):
  composite_seed_str = ",".join(map(str, seeds))
  label_file = os.path.join(get_prompt_dir(COMPOSITE_DIR, shape, model_type, prompt),
                            "graphcut",
                            "comp_{}".format(composite_seed_str),
                           "gc_{}_{}_seeds_{}.{}".format(
                            header, shape, composite_seed_str, suffix))
  return label_file

def get_vanilla_image_file(shape, model_type, prompt, seed):
  fn = os.path.join(VANILLA_DIR,
                      shape,
                      model_type,
                      prompt.replace(" ", "_"),
                      "seed{}.png".format(seed))
  return fn

def compute_graph_cut(shape, model_type, prompts, seeds, mask_suffix=None):
  if len(prompts) == 1:
    prompts = prompts * len(seeds)
  elif len(prompts) > len(seeds):
    prompts = prompts[:len(seeds)]

  metadata={}
  metadata['seeds'] = seeds
  metadata['output_dir'] = get_prompt_dir(COMPOSITE_DIR, shape, model_type, prompts[0])
  metadata['shape'] = shape

  # Load in input images, K feature files, and stroke files
  images = [] # Original input images
  ks = [] # K feature files
  s_files = [] # Stroke files
  for i, seed in enumerate(seeds):
    img_file = get_vanilla_image_file(shape, model_type, prompts[i], seed)
    images.append(img_file)

    k_file = os.path.join(get_prompt_dir(VANILLA_DIR, shape, model_type, prompts[i]),  
                          "seed={}".format(seed), 
                          "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor.pth")
    ks.append(k_file)

    suffix = None
    if mask_suffix is not None:
        suffix = mask_suffix[i]
    s_file = get_mask_file(shape, model_type, prompts[i], seed, mask_suffix=suffix)
    s_files.append(s_file)

  h, w, c = np.array(Image.open(images[0])).shape
  size = (h // 8, w // 8)

  _, _, _, _, _, vis_files = multi_graph_cut(metadata, images, ks, s_files, size)
  save_graphcut_masks(shape, model_type, prompts, seeds)

  return vis_files

# Save graph cut masks
def save_graphcut_masks(shape, model_type, prompts, seeds):
  label_file = get_gc_file(shape, model_type, prompts[0], seeds)

  graphcut_dir = os.path.join(get_prompt_dir(COMPOSITE_DIR, shape, model_type, prompts[0]),
                              "graphcut",
                              "comp_{}".format(",".join(map(str, seeds))))
  os.makedirs(graphcut_dir, exist_ok=True)

  result_graph = np.load(label_file)

  for i, seed in enumerate(seeds):
    mask = np.zeros_like(result_graph, dtype=np.uint8)
    mask[result_graph == i] = 255
    
    # Save graph cut mask to masks folder.
    fn = get_mask_file(shape, model_type, prompts[i], seed, gc=True)
    Image.fromarray(mask).save(fn)

    # Also save a copy of graph cut mask to graphcut directory
    fn = os.path.join(graphcut_dir, "gc_mask_{}_seed_{}.png".format(i, seed))
    Image.fromarray(mask).save(fn)

def get_composite_image_file(shape, model_type, prompt, composite_seed_str):
  return os.path.join(COMPOSITE_DIR,
                      shape,
                      model_type,
                      prompt.replace(" ", "_"),
                      "graphcut",
                      "comp_{}".format(composite_seed_str),
                      "gc_composite_{}_seeds_{}.png".format(shape, composite_seed_str))

def log_composite_info(shape, model_type, prompts, seeds, timestamp_str):
  composite_seeds_str = ",".join(map(str, seeds))
  log_dir = os.path.join(get_prompt_dir(COMPOSITE_DIR, shape, model_type, prompts[0]), "logs", timestamp_str)
  os.makedirs(log_dir, exist_ok=True)

  # Log composed image
  comp_fn = get_composite_image_file(shape, model_type, prompts[0], composite_seeds_str)
  _, basename = os.path.split(comp_fn)
  newname = "{}-{}".format(timestamp_str, basename)
  new_fn = os.path.join(log_dir, newname)
  shutil.copyfile(comp_fn, new_fn)

  # Log original images
  for i in range(len(seeds)):
    s, p = seeds[i], prompts[i]
    seed_img_fn = get_vanilla_image_file(shape, model_type, p, s)
    _, basename = os.path.split(seed_img_fn)

    # Append prompt if it's different from the base image's prompt
    different_prompt = "" if p == prompts[0] else "-{}".format(p.replace(" ", "_"))  
    newname = "{}-{}{}.{}".format(timestamp_str, basename.split(".")[0], different_prompt, basename.split(".")[1])
    new_fn = os.path.join(log_dir, newname)
    shutil.copyfile(seed_img_fn, new_fn)

  for i, s in enumerate(seeds):
    # Copy over graph cut mask to logs
    fn = get_mask_file(shape, model_type, prompts[i], s, gc=True)
    _, basename = os.path.split(fn)
    newname = "{}-comp_{}-mask-{}-seed-{}.png".format(timestamp_str, composite_seeds_str, i, s)
    new_fn = os.path.join(log_dir, newname)
    shutil.copyfile(fn, new_fn)

    # Copy over original stroke mask as well
    fn = get_mask_file(shape, model_type, prompts[i], s, gc=False)
    _, basename = os.path.split(fn)
    newname = "{}-comp_{}-mask-{}-seed-{}.png".format(timestamp_str, composite_seeds_str, i, s)
    new_fn = os.path.join(log_dir, newname)
    shutil.copyfile(fn, new_fn)

  # Log graph cut strokes.
  prompt = prompts[0].replace(" ", "_")  # Prompt of base image.
  for i in range(len(seeds)):
    fn = get_gc_file(shape, model_type, prompt, seeds, header="stroke_{}".format(i), suffix='png')
    _, basename = os.path.split(fn)

    newname = "{}-{}".format(timestamp_str, basename)
    new_fn = os.path.join(log_dir, newname)
    shutil.copyfile(fn, new_fn)

# Get the stack of masks. Set use_gc_masks=True to get graph cut masks.
def get_image_stack_masks(shape, model_type, prompts, seeds, use_gc_masks=True):
  mask_files = []
  for i, s in enumerate(seeds):
    if i > 0:
      fn = get_mask_file(shape, model_type, prompts[i], s, gc=use_gc_masks)
      if os.path.isfile(fn):
        mask_files.append(fn)  
  return mask_files


def get_qkv_dir(shape, model_type, prompt, seed):
  qkv_inject_dir = os.path.join(shape,
                                 model_type,
                                 prompt.replace(" ", "_"),
                                 "seed={}".format(seed))
  return qkv_inject_dir