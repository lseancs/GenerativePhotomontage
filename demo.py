import gradio as gr
import numpy as np
from PIL import Image
import os
from diffusers.utils import load_image
from run_generative_photomontage import *

def run_vanilla_inference(
    model_type,
    shape,
    prompt,
    initial_seed,
    num_seeds,
    save_QKV_features):
  cond_img_file = "./data/{}-{}.png".format(shape, model_type)
  output_qkv_dir = os.path.join(shape, model_type)if save_QKV_features else None
  
  return inference_gpm(
            model_type, 
            os.path.join(VANILLA_DIR, shape), 
            cond_img_file, 
            prompt, 
            initial_seed,
            num_seeds, 
            inject_self=False,  # Set to False to run vanilla ControlNet. Set to True to enable attention injection.
            qkv_save_folder=output_qkv_dir)

def run_gen_photomontage(
    model_type,
    shape, 
    prompt1=None, 
    prompt2=None, 
    prompt3=None,
    prompt4=None, 
    prompt5=None, 
    seed1=None, 
    seed2=None, 
    seed3=None, 
    seed4=None, 
    seed5=None):
  seeds = [seed1, seed2, seed3, seed4, seed5]
  seeds = [s for s in seeds if s is not None and s != ""]
  prompts = [prompt1, prompt2, prompt3, prompt4, prompt5]
  prompts = prompts[:len(seeds)]
  return generate_photomontage(model_type, shape, prompts, seeds)

def run_graph_cut(shape, model_type,
                      prompt1, prompt2, prompt3, prompt4, prompt5,
                      seed1, seed2, seed3, seed4, seed5):
  seeds = [seed1, seed2, seed3, seed4, seed5]
  seeds = [s for s in seeds if s is not None and s != ""]
  prompts = [prompt1, prompt2, prompt3, prompt4, prompt5]
  prompts = prompts[:len(seeds)]

  vis_files = compute_graph_cut(shape, 
                                model_type, 
                                prompts, 
                                seeds)
  return [vis_files['label']], [vis_files['pixel_blend']]


def load_image_with_strokes(shape, model_type, prompt, seed):
  layer = None
  mask = get_mask_file(shape, model_type, prompt, seed)

  if os.path.isfile(mask):
    layer = np.array(load_image(mask).convert('RGBA'))
    layer[:, :, 3] = 0
    masked = np.any(layer[:, :, :3] > 0, axis=2)
    layer[masked, 3] = 255
    layer[masked, :3] = 255
    layer = [Image.fromarray(layer)]

  img = get_vanilla_image_file(shape, model_type, prompt, seed)

  return {"background": img, "layers": layer, "composite": img}

def save_stroke(editor, shape, model_type, prompt, seed):
  i, layer = 0, editor["layers"][0]
  layer = layer.convert("L")
  mask_file = get_mask_file(shape, model_type, prompt, seed)
  layer.save(mask_file)
  return "Saved stroke to {}".format(mask_file)

def save_sketch(sketch_name, sketch_type, sketch_editor):
  sketch_fn = "{}-{}.png".format(sketch_name, sketch_type)
  full_sketch_fn = os.path.join("data", sketch_fn)

  background = sketch_editor["composite"]
  background.save(full_sketch_fn)

  choices = get_shape_choices()
  updated_dropdown = gr.Dropdown(choices = choices, label="Input Condition")
  updated_dropdown2 = gr.Dropdown(choices = choices, label="Input Condition")
  
  return ["Saved image to {}. \nTo use it in Step 1, select \"{}\" for ControlNet Pretrained Model and \"{}\" for Input Condition.".format(full_sketch_fn, sketch_type, sketch_name), 
          updated_dropdown, updated_dropdown2]

def get_shape_choices():
  choices = set()
  for fn in os.listdir("data"):
    if fn.endswith("png"):
      shape = fn.split(".png")[0]
      choices.add(shape.split("-")[0])
  return sorted(list(choices))

### UI Code ###
with gr.Blocks() as demo:
    gr.Markdown("# Generative Photomontage Demo")
    gr.Markdown("### Step 0 (Optional): Create input condition for Vanilla ControlNet. Alternatively, you can use one of the provided input conditions in Step 1.")
    gr.Markdown("Here, you can upload a depth/edge map or draw a sketch. Make sure to designate the condition type and click save.")

    with gr.Row():
      sketchEditor = gr.ImageEditor(type="pil", image_mode="RGB", show_download_button = True, brush=gr.Brush(colors=["#FFFFFF"]))

    with gr.Row():
      sketch_type = gr.Dropdown(label="ControlNet Condition Type", choices=["canny", "scribble", "hed", "depth", "openpose"], value="scribble")
      sketch_fn = gr.Textbox(label="Condition name", value="my_input_sketch")
    
    with gr.Column():
      save_sketch_button = gr.Button("Save")
      save_sketch_label = gr.Label(label="")

    gr.Markdown("# Step 1: Run Vanilla ControlNet")
    gr.Markdown("### Note: You must \"Save QKV features\" for images that you wish to composite later. The size of QKV feature files depends on the image resolution.")
    gr.Markdown("For an image resolution of 512x512, the QKV feature files are ~2GB per image.")

    input_condition_choices = get_shape_choices()
    with gr.Row():
      vanilla_prompt = gr.Textbox(label="Prompt", value="A robot from the future")
      vanilla_model = gr.Dropdown(label="ControlNet Pretrained Model", choices=["canny", "scribble", "hed", "depth", "openpose"], value="scribble")
      vanilla_shape = gr.Dropdown(label="Condition Name", choices=input_condition_choices, value="robot")
      
    with gr.Row():
      init_seed = gr.Number(label="Initial seed", value=0)
      num_seeds = gr.Number(label="Number of images", value=1)      
      save_QKV_features = gr.Checkbox(label="Save QKV Features", value=True)

    vanilla_inference_btn = gr.Button("Run Vanilla ControlNet")

    with gr.Row():
      vanilla_cond = gr.Gallery(label="Conditioned Image", show_label=True, height=500, preview=True)
      vanilla_output = gr.Gallery(label="Output Images", show_label=True, preview=True)

    vanilla_inference_btn.click(fn=run_vanilla_inference, 
                                inputs=[vanilla_model, vanilla_shape, vanilla_prompt, init_seed, num_seeds, save_QKV_features], 
                                outputs=[vanilla_cond, vanilla_output])
    
    gr.Markdown("# Step 2: Let's composite images!")
    gr.Markdown("## Step 2.1: Select images to composite. You may leave fields blank if compositing fewer than 5 images.")

    with gr.Row():
      composite_model = gr.Dropdown(label="ControlNet Pretrained Model", choices=["canny", "scribble", "hed", "depth", "openpose"], value="scribble")
      input_condition_choices = get_shape_choices()
      composite_shape = gr.Dropdown(label="Condition Name", choices=input_condition_choices, value="robot")

    with gr.Row():
      prompt1 = gr.Textbox(label="Prompt 1", value="A robot from the future")
      prompt2 = gr.Textbox(label="Prompt 2", value="A robot from the future")
      prompt3 = gr.Textbox(label="Prompt 3", value="A robot from the future")
      prompt4 = gr.Textbox(label="Prompt 4", value="")
      prompt5 = gr.Textbox(label="Prompt 5", value="")

    with gr.Row():
      seed1 = gr.Number(label="Seed 1 (Background seed)", value=0)
      seed2 = gr.Number(label="Seed 2", value=4)    
      seed3 = gr.Number(label="Seed 3", value=3)
      seed4 = gr.Number(label="Seed 4", value="")
      seed5 = gr.Number(label="Seed 5", value="")

    gr.Markdown("## Step 2.2: For each image, select desired regions with brush strokes. Make sure you have saved a stroke per image.")

    gr.Markdown("#### Load the image")

    with gr.Row():
      input_condition_choices = get_shape_choices()

      load_model = gr.Dropdown(label="ControlNet Pretrained Model", choices=["canny", "scribble", "hed", "depth", "openpose"], value="scribble")
      load_shape = gr.Dropdown(label="Condition Name", choices=input_condition_choices, 
                            value="robot")
      load_prompt = gr.Textbox(label="Load Prompt", value="A robot from the future")
      load_seed = gr.Number(value=0, label="Load Seed")
    
    with gr.Row():
      btn_load = gr.Button("Load Image")

    gr.Markdown("#### Draw strokes below. Click \"Update Canvas\" if you run into UI issues when drawing strokes. IMPORTANT: Click \"Save Stroke\" when done.")

    with gr.Row():
      stroke_editor = gr.ImageEditor(type="pil", image_mode="RGB", brush=gr.Brush(colors=["#FFFFFF"]))

    save_sketch_button.click(save_sketch, inputs=[sketch_fn, sketch_type, sketchEditor], outputs=[save_sketch_label, vanilla_shape, load_shape])
    btn_load.click(load_image_with_strokes, inputs=[load_shape, load_model, load_prompt, load_seed], outputs=[stroke_editor])

    with gr.Row():
      clear_strokes_btn = gr.Button("Clear All Strokes")
      update_stroke_editor_btn = gr.Button("Update Canvas")
      save_stroke_btn = gr.Button("Save Stroke")
    
    # Message box
    save_stroke_label = gr.Label(label="")

    def update():
      gr.update()

    def reset(shape, model_type, prompt, seed):
      img = get_vanilla_image_file(shape, model_type, prompt, seed)
      return {"background": img, "layers": [], "composite": img}

    clear_strokes_btn.click(reset, inputs=[load_shape, load_model, load_prompt, load_seed], outputs=stroke_editor)
    update_stroke_editor_btn.click(update)
    save_stroke_btn.click(save_stroke, 
                          inputs=[stroke_editor, load_shape, load_model, load_prompt, load_seed],
                          outputs=[save_stroke_label])

    gr.Markdown("## Step 2.3: Click \"Graph cut\" to segment the images based on the saved strokes above.")
    gr.Markdown("Graph cut results are visualized below (left: feature space; right: image space visualization).")
    gr.Markdown("You can change the graph cut by going back to Step 2.2 and iterating on brush strokes.")

    with gr.Row():
      graphcut_btn = gr.Button("Graph cut")

    with gr.Row():
      gc_features_preview = gr.Gallery(value=None, label="Graph cut results (feature space)", show_label=True, preview=True)
      gc_pixels_preview = gr.Gallery(value=None, label="Graph cut results visualized in pixel space", show_label=True, preview=True)

    graphcut_btn.click(run_graph_cut, inputs=[composite_shape, composite_model, 
                                                  prompt1, prompt2, prompt3, prompt4, prompt5,
                                                  seed1, seed2, seed3, seed4, seed5], 
                                          outputs=[gc_features_preview, gc_pixels_preview])

    gr.Markdown("## Step 2.4: Composition")
    gr.Markdown("Once you're satisfied the graph cut, click on \"Composite\" to get the final composite!")
    composite_btn = gr.Button("Composite")

    with gr.Row():
      composite_input_cond = gr.Gallery(label="Conditioned Image", show_label=True, height=500, preview=True)
      composite_output = gr.Gallery(label="Output Composite Images", show_label=True, preview=True)

    composite_btn.click(fn=run_gen_photomontage, 
                        inputs=[composite_model, composite_shape, 
                                prompt1, prompt2, prompt3, prompt4, prompt5,
                                seed1, seed2, seed3, seed4, seed5], 
                        outputs=[composite_input_cond, composite_output])

if __name__ == "__main__":  
  demo.launch(server_name='0.0.0.0', server_port=7800)