# Generative Photomontage
arXiv | Project Website

## Setup

Tested on CUDA 12.0 with Ubuntu 20.04.6 LTS. 
Python 3.9, Diffusers 0.22.0. GPU: NVIDIA A10.

Step 1: Environment setup
```
conda create -n gpm python=3.9.18 numpy=1.25.2
conda activate gpm
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch -c nvidia
pip install -r requirements.txt
```

Step 2: Install graph cut solver:
```
git clone https://github.com/amueller/gco_python
cd gco_python && make
pip install -e .
```

To check if the install was successful, run:
```
python example.py 
```
You should see a 2D matrix (3x3) printed out with no error.

Step 3: Run a simple example
```
sh run_examples.sh 
```
To try out more examples from the paper, you can uncomment additional lines in the script. 

Before running the additional examples, first download the stroke masks [here](https://drive.google.com/file/d/1IkDyBfyfMxi9Qj_U9asNnJknibrPqkKL/view?usp=sharing).
In the root directory, unzip the files:
```
unzip generative_photomontage_data.zip
```

## Demo

To run the demo:
```
python demo.py
```
Go to: http://localhost:7800

## Detailed Command Line Instructions:
1. Run `vanilla.py` to run vanilla ControlNet and store QKV feature vectors.
```
python run_vanilla.py --cond data/robot-scribble.png --prompts "A robot from the future" --seeds 0 4 3
```
You only need to run this script ONCE for each vanilla image.
The script will generate and store images in `vanilla/{cond_name}/{model_name}/{prompt}/`.
The corresponding QKV features will be stored in `vanilla/{cond_name}/{model_name}/{prompt}/seed-{SEED}/`.

2. Run `run_generative_photomontage.py` to composite the images based on user strokes.
First, place stroke masks into the corresponding folder: `composite/{cond_name}/{model_name}/{prompt}/masks/mask_{SEED}.png`. 

Then run:
```
python run_generative_photomontage.py --cond data/robot-scribble.png --prompts "A robot from the future" --seeds 0 4 3
```
Note: You can create stroke masks with the demo (above), which has a UI for drawing sketches.