#!/bin/sh

# Run "run_vanilla.py" to get vanilla ControlNet outputs and to save QKV features. You only need to run this once for each image.
# Run "run_generative_photomontage.py" to run graph cut and composite.
# Strokes should be saved in composite/{COND_NAME}/{MODEL}/{PROMPT}/masks/
for script in "run_vanilla.py" "run_generative_photomontage.py"
do
    python ${script} --cond data/robot-scribble.png --prompts "A robot from the future" --seeds 0 4 3

    # Uncomment to try other examples from paper
    # python ${script} --cond data/dog-canny.png --prompts "A dog on ice" "A dog on grass" --seeds 4 8
    # python ${script} --cond data/applelogo-canny.png --prompts "A rock on grass" --seeds 1 5 6
    # python ${script} --cond data/bird-canny.png --prompts "A bird looking out into the wilderness" --seeds 4 2 6
    # python ${script} --cond data/building-depth.png --prompts "A Japanese-style stone house" --seeds 7 5 9 0
    # python ${script} --cond data/fairies-canny.png \
    #         --prompts "Fairies sitting in a brown boat" \
    #                     "Green fairies sitting in a boat" \
    #                     "Green fairies sitting in a boat" \
    #                     "Red fairies sitting in a boat" \
    #                     "Blue fairies sitting in a boat" \
    #             --seeds 2 4 5 1 5
    # python ${script} --cond data/waffle-scribble.png --prompts "A waffle pancake" --seeds 5 8
    # python ${script} --cond data/dancer-openpose.png --prompts "A dancer" --seeds 9 8
    # python ${script} --cond data/snake-scribble.png --prompts "A colorful snake on a branch" --seeds 2 6 0
    # python ${script} --cond data/temple-scribble.png --prompts "A futuristic temple" --seeds 2 8 6
    # python ${script} --cond data/red_riding_hood-hed.png \
    #                 --prompts "Little red riding hood walking towards a red and green house" \
    #                         "A blue meadow" \
    #                         "A blue meadow" \
    #                         "Yellow door" \
    #             --seeds 4 1 4 0

    # if [ $script = "run_vanilla.py" ]; then
    #     python ${script} --cond data/dog-canny.png --prompts "A dog on grass" --seeds 3 8
    #     python ${script} --cond data/bird-canny.png --prompts "A bird looking out into the wilderness" --seeds 4 2 9
    # else
    #     # Specify a mask file suffix to select between multiple masks (for the same image).
    #     python ${script} --cond data/dog-canny.png --prompts "A dog on grass" --seeds 3 8 --mask_suffix . alt
    #     python ${script} --cond data/bird-canny.png --prompts "A bird looking out into the wilderness" --seeds 4 2 9 --mask_suffix alt . .
    # fi
done