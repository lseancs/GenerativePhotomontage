import os
import torch
import matplotlib.pyplot as plt
from attention_processor_gpm import *
from pygco import cut_simple_vh
from sklearn.decomposition import PCA
from PIL import Image

def load_and_prepare_image(fn):
    img = Image.open( fn )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def resize_img(stroke, size, mode=transforms.InterpolationMode.NEAREST):
    stroke = Image.fromarray(stroke)
    after = np.array(transforms.functional.resize(stroke, size, mode), dtype=np.uint8)

    return after

def load_k(fn):
    all = torch.load(fn, map_location='cuda')
    k = all[19]['k'].cpu().numpy()
    return k

def compute_features_pca(features, height, n_components=10):  # Q has shape (B, heads, H*W, dim)
    final_features = []
    for i in range(features.shape[0]):
        f = features[i] # heads x (H*W) x dim
        f = np.swapaxes(f, 0, 1)  # (H*W) x heads x dim
        f = f.reshape(f.shape[0], -1)  # (H*W) x (heads x dim)

        pca = PCA(n_components=n_components)
        pca.fit(f)
        features_pca = pca.transform(f)
        features_pca = features_pca.reshape(height, -1, n_components)  # H * W x 10
        final_features.append(features_pca)

    return np.stack(final_features, axis=0) # B x H x W x n_components

def multi_graph_cut(metadata, images, k_feature_files, stroke_mask_files, size, sigma=10, n_components=10):
    plt.close()

    # Load and prepare images
    input_imgs = []
    for img in images:
        input_imgs.append(load_and_prepare_image(img))
    
    # Load in prepare K features
    k_feature_vectors = []
    for fn in k_feature_files:
        k_feature_vectors.append(load_k(fn))

    # Load in prepare user stroke masks
    input_stroke_masks = []
    resized_stroke_masks = []
    for stroke_fn in stroke_mask_files:
        smask = load_and_prepare_image(stroke_fn)
        input_stroke_masks.append(smask)
        resized_stroke_masks.append(resize_img(smask, size))

    # create unary terms
    unaries = generate_unary_constraints(resized_stroke_masks)

    # Only apply edge weights (below) if neighboring labels are different
    pairwise = generate_pairwise_constraints(len(k_feature_vectors))

    # create edge weights of neighboring features
    horz, vert = generate_edge_weights(k_feature_vectors,  size=size, sigma=sigma, n_components=n_components)

    result_graph = cut_simple_vh(unaries, pairwise, vert, horz, n_iter=100)

    stroke_images, _, label_img_file, pixel_composite_file = save_result_images(
        result_graph, input_imgs, input_stroke_masks, metadata)

    vis_files = {}
    vis_files['stroke_images'] = stroke_images
    vis_files['label'] = label_img_file
    vis_files['pixel_blend'] = pixel_composite_file

    return result_graph, unaries, pairwise, horz, vert, vis_files

# Saves a bunch of result files and visualizations.
def save_result_images(result_graph, images, stroke_masks, metadata):
    # Resize features space graph cut to image space.
    result_graph_highres = resize_img(result_graph, images[0].shape[0:2])

    # Create pixel composite based on resized graph cut results.
    composite = np.array(images[0], dtype=np.uint8)
    for i, img in enumerate(images):
        if i == 0:  # base image
            continue
        composite[result_graph_highres == i, :] = img[result_graph_highres==i, :]

    indices_str = ",".join(map(str, metadata['seeds']))
    imgs_with_strokes = []
    stroke_image_fns = []
    plt.axis('off')
    subdir = os.path.join(metadata['output_dir'], "graphcut", "comp_{}".format(indices_str))
    os.makedirs(subdir, exist_ok=True)

    for img, smask in zip(images, stroke_masks):
        img_with_stroke = np.array(img)
        img_with_stroke[smask != 0, :] = 255
        imgs_with_strokes.append(img_with_stroke)

        plt.figure()
        plt.imshow(img_with_stroke, interpolation='nearest')
        plt.axis('off')
        fn = os.path.join(metadata['output_dir'],
                          "graphcut",
                          "comp_{}".format(indices_str),
                          "gc_stroke_{}_{}_seeds_{}.png".format(len(stroke_image_fns), metadata['shape'], indices_str))
        plt.savefig(fn, bbox_inches='tight',pad_inches = 0)
        stroke_image_fns.append(fn)

    # Saving labels output to .npy
    labels_npy_file = os.path.join(metadata['output_dir'],
                            "graphcut",
                            "comp_{}".format(indices_str),
                            "gc_label_{}_seeds_{}.npy".format(metadata['shape'], indices_str))
    np.save(labels_npy_file, result_graph)

    # Saving label image
    plt.figure()
    label_img_file = os.path.join(metadata['output_dir'],
                             "graphcut",
                             "comp_{}".format(indices_str),
                           "gc_label_{}_seeds_{}.png".format(metadata['shape'], indices_str))
    plt.imshow(result_graph[:, :, None], interpolation='nearest')
    plt.tight_layout(pad=0.5)
    plt.axis('off')
    plt.gca().get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
    plt.gca().get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis
    plt.savefig(label_img_file, bbox_inches='tight',pad_inches = 0)

    # Saving pixel composite
    pixel_composite_file = os.path.join(metadata['output_dir'],
                             "graphcut",
                             "comp_{}".format(indices_str),
                             "gc_composite_{}_seeds_{}.png".format(metadata['shape'], indices_str))
    Image.fromarray(composite).save(pixel_composite_file)

    return stroke_image_fns, labels_npy_file, label_img_file, pixel_composite_file

def generate_unary_constraints(stroke_masks, penalty=1e6):
    h, w = stroke_masks[0].shape
    unaries = np.zeros((h, w, len(stroke_masks)), dtype=np.int32)

    # High penalty for choosing other image at stroke pixels
    for i, mask in enumerate(stroke_masks):
        for c in range(len(stroke_masks)):
            if i != c:
                unaries[mask == 255, c] = penalty
        
    return unaries

def generate_pairwise_constraints(num_labels=2):
    # Only apply edge weights (below) if neighboring labels are different
    # Create an array of size (2, 2): 1 for different labels. Use np.int32.
    pairwise = np.ones((num_labels, num_labels), dtype=np.int32) - np.eye(num_labels, dtype=np.int32)
    return pairwise

def exponential_diff(val1, val2, sigma=10):
    diff = np.abs(val1 - val2)
    acc = np.sum(diff, axis=2)
    exp = 100*np.exp(-acc / (2*sigma))
    return exp

def generate_edge_weights(feature_vectors, size, sigma=10, n_components=10):
    height, _ = size
    pca_features = []

    for vec in feature_vectors:
        vec = compute_features_pca(vec, height, n_components=n_components) # batch x H x W x n_components
        vec = np.moveaxis(vec, (0, 1, 2), (2, 0, 1)) # H x W x batch x n_components
        vec = vec.reshape(vec.shape[0], vec.shape[1], -1)
        pca_features.append(vec)

    h, w = pca_features[0].shape[:2]
    horz = np.zeros((h, w), dtype=np.int32)
    vert = np.zeros((h, w), dtype=np.int32)

    diff_h, diff_v = 0, 0
    for vec in pca_features:
        diff_h1 = exponential_diff(vec[:, :-1], vec[:, 1:], sigma)
        diff_v1 = exponential_diff(vec[:-1, :], vec[1:, :], sigma)

        diff_h += diff_h1
        diff_v += diff_v1

    horz[:, :-1] = diff_h
    vert[:-1, :] = diff_v

    return horz, vert
