
from diffusers.models.attention_processor import *
import math
import torch
import os
import numpy as np
import torchvision.transforms as transforms
from diffusers.utils import load_image

VANILLA_DIR = "vanilla"
COMPOSITE_DIR = "composite"

class GPMAttentionProcessor:
    r"""
    Attention Processor for Generative Photomontage.
    """

    def __init__(self, 
                 name, 
                 control_image, 
                 num_inference_steps, 
                 qkv_save_dir=None,
                 inject_after_timestep=0, 
                 inject_layers='all',
                 inject_self=True,
                 inject_qkv_dirs=None,
                 injection_mask_files=None):
        
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.name = name
        self.control_image = control_image
        self.aspect_ratio = control_image.width / control_image.height
        self.register_latent_shape()

        self.num_inference_steps = num_inference_steps
        self.inject_after_timestep = inject_after_timestep
        self.inject_layers = inject_layers

        # Set to False to avoid any self-attention injection. If True, will inject according to inject_layers and inject_after_timestep.
        self.inject_self = inject_self

        # For saving QKV features
        self.qkv_save_dir = qkv_save_dir
        self.kqv = {}
        
        # For injecting QKV features
        self.inject_qkv_dirs = inject_qkv_dirs
        self.injection_mask_files = injection_mask_files  # Binary mask files of regions to inject (obtained from graph cut)
        self.inject_binary_masks = None  # Binary masks
        self.qkv_feature_filenames = None
        self.read_binary_masks()
        
        self.step = 0
        self.seed=None
        self.component = None

        # Used to keep QKV in RAM
        self.q_loaded_vectors = {}
        self.k_loaded_vectors = {}
        self.v_loaded_vectors = {}

        if self.qkv_save_dir is not None:
            subfolder = os.path.join(VANILLA_DIR, self.qkv_save_dir)
            os.makedirs(subfolder, exist_ok=True)
            self.qkv_save_dir = subfolder

    def register_latent_shape(self):
        h, w, _ = np.array(self.control_image).shape
        self.init_latent_shape = (h // 8, w // 8)

    def set_component(self, c):
        self.component = c

    def set_seed(self, seed):
        self.seed=seed

        if self.inject_qkv_dirs is not None and 'attn1' in self.name:
            self.qkv_feature_filenames = []
            for qkv_dir in self.inject_qkv_dirs:
                filepath = os.path.join(VANILLA_DIR, qkv_dir, "{}.pth".format(self.name))
                if not os.path.isfile(filepath):
                    filepath = os.path.join(VANILLA_DIR, qkv_dir, self.prompt.replace(" ", "_"), "seed={}".format(self.seed), "{}.pth".format(self.name))
                
                assert os.path.isfile(filepath), "{} does not exist.".format(filepath)
                self.qkv_feature_filenames.append(filepath)
            # print("Going to use QKV features from {}".format(self.qkv_feature_filenames))

    def set_prompt(self, prompt):
        self.prompt = prompt

    @property
    def is_cross(self) -> bool:
        return "attn2" in self.name
    
    def read_binary_masks(self):
        if self.injection_mask_files is not None:
            self.inject_binary_masks = []
            for mask in self.injection_mask_files:
                # Load in the binary mask
                raw_mask = load_image(mask)
                raw_mask = transforms.functional.pil_to_tensor(raw_mask.convert('L'))
                copy = torch.zeros_like(raw_mask).to(torch.bool)
                copy[raw_mask>0] = 1
                mask = copy.to(torch.bool)[0]
                self.inject_binary_masks.append(mask)

    @property
    def inject(self):
        attn_condition = self.inject_self and 'attn1' in self.name
        time_condition = self.step >= self.inject_after_timestep * self.num_inference_steps
        if self.inject_layers == 'all':
            layer_condition = True
        elif self.inject_layers == 'none':
            layer_condition = False
        else:
            layers = self.inject_layers.split(",")
            layer_condition = False
            for layer in layers:
                block_type = layer[0].lower()
                layer_num = int(layer[1]) if len(layer) > 1 else None
                if block_type == 'd':
                    block_type = "down_blocks.{}".format(layer_num)
                elif block_type == 'u':
                    block_type = "up_blocks.{}".format(layer_num)
                else:
                    block_type = "mid_block"
                if block_type in self.name:
                    layer_condition = True
                    break
        return time_condition and layer_condition and attn_condition

    def finish(self):
        self.step = 0
        self.save_kqv()

        self.output_dir = None
        self.prompt = None

        self.q_loaded_vectors = None
        self.k_loaded_vectors = None
        self.v_loaded_vectors = None

    def save_kqv(self):
        if self.is_cross or self.qkv_save_dir is None:
            return
        assert self.seed is not None, "Seed should not be None."
        kqv_subdir=os.path.join(self.qkv_save_dir, self.prompt.replace(" ", "_"), "seed={}".format(self.seed))
        kqv_filepath = os.path.join(kqv_subdir, "{}.pth".format(self.name))
        os.makedirs(kqv_subdir, exist_ok=True)
        torch.save(self.kqv, kqv_filepath)
        # print("Saved QKV to {}".format(kqv_filepath))
        self.kqv = {}

    @property
    def height(self) -> int:
        if self.aspect_ratio < 1: # width less than height:
            return -1
        block = self.name.split("_")[0]
        if block == "down":
            layer = int(self.name.split(".")[1])
            return int(math.ceil(self.init_latent_shape[0] / math.pow(2, layer)))
        elif block == "mid":
            return int(math.ceil(self.init_latent_shape[0] / math.pow(2, 3)))
        else:
            layer = int(self.name.split(".")[1])
            return int(math.ceil(self.init_latent_shape[0] / math.pow(2, 3-layer)))
    
    @property
    def width(self) -> int:
        if self.aspect_ratio > 1: # height less than width:
            return -1
        block = self.name.split("_")[0]
        if block == "down":
            layer = int(self.name.split(".")[1])
            return int(math.ceil(self.init_latent_shape[1] / math.pow(2, layer)))
        elif block == "mid":
            return int(math.ceil(self.init_latent_shape[1] / math.pow(2, 3)))
        else:
            layer = int(self.name.split(".")[1])
            return int(math.ceil(self.init_latent_shape[1] / math.pow(2, 3-layer)))

    def get_query(self, index=0):
        if self.is_cross or self.qkv_feature_filenames is None:
            return None
        if index not in self.q_loaded_vectors:
            all = torch.load(self.qkv_feature_filenames[index], map_location='cpu')
            self.q_loaded_vectors[index] = torch.stack([all[i]['q'] for i in range(self.num_inference_steps)])
            self.k_loaded_vectors[index] = torch.stack(tuple([all[i]['k'] for i in range(self.num_inference_steps)]))
            self.v_loaded_vectors[index] = torch.stack(tuple([all[i]['v'] for i in range(self.num_inference_steps)]))
        return self.q_loaded_vectors[index][self.step].half().cuda()
    
    def get_key(self, index=0):
        if self.is_cross or self.qkv_feature_filenames is None:
            return None
        if index not in self.k_loaded_vectors:
            all = torch.load(self.qkv_feature_filenames[index], map_location='cpu')
            self.q_loaded_vectors[index] = torch.stack([all[i]['q'] for i in range(self.num_inference_steps)])
            self.k_loaded_vectors[index] = torch.stack(tuple([all[i]['k'] for i in range(self.num_inference_steps)]))
            self.v_loaded_vectors[index] = torch.stack(tuple([all[i]['v'] for i in range(self.num_inference_steps)]))
        return self.k_loaded_vectors[index][self.step].half().cuda()

    def get_value(self, index=0):
        if self.is_cross or self.qkv_feature_filenames is None:
            return None
        if index not in self.v_loaded_vectors:
            all = torch.load(self.qkv_feature_filenames[index], map_location='cpu')
            self.q_loaded_vectors[index] = torch.stack([all[i]['q'] for i in range(self.num_inference_steps)])
            self.k_loaded_vectors[index] = torch.stack(tuple([all[i]['k'] for i in range(self.num_inference_steps)]))
            self.v_loaded_vectors[index] = torch.stack(tuple([all[i]['v'] for i in range(self.num_inference_steps)]))
        return self.v_loaded_vectors[index][self.step].half().cuda()

    def inject_image_stack_vector(self, vector, injected_vectors):
        assert isinstance(self.inject_binary_masks, list), "There is no list: {}".format(self.inject_binary_masks)
        assert len(self.qkv_feature_filenames) == len(self.inject_binary_masks) + 1, "Number of Q features {} do not match masks {}".format(len(self.qkv_feature_filenames), len(self.inject_binary_masks))

        orig_vector = vector.clone().detach()
        vector = vector.view(vector.shape[0], vector.shape[1], self.height, self.width, vector.shape[-1]).to('cuda')
        for i in range(len(self.inject_binary_masks)):
            injected_vector = injected_vectors[i]

            # Reshape into batch, heads, height, width, hidden before applying injection
            injected_vector = injected_vector.view(injected_vector.shape[0], injected_vector.shape[1], self.height, self.width, injected_vector.shape[-1]).to('cuda')

            # Resize binary mask to this layer's dimensions.
            resized_mask = transforms.functional.resize(self.inject_binary_masks[i].unsqueeze(0), vector.shape[2:4], transforms.InterpolationMode.NEAREST)[0]
            resized_mask = resized_mask.to('cuda')
            vector[:, :, resized_mask, :] = injected_vector[:, :, resized_mask, :]
        
        vector = vector.view(vector.shape[0], vector.shape[1], -1, vector.shape[-1])
        return orig_vector, injected_vectors, vector

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale) 

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # batch, heads, pixels, channels per head
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # batch, heads, tokens, channels per head
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if not self.is_cross and self.qkv_save_dir is not None:
            self.kqv[self.step] = {}
            self.kqv[self.step]['q'] = query
            self.kqv[self.step]['k'] = key
            self.kqv[self.step]['v'] = value

        if self.inject and self.qkv_feature_filenames is not None:
            injected_query = self.get_query()

            if injected_query is not None:
                injected_vectors = []
                for i in range(1, len(self.qkv_feature_filenames)):
                    injected_vectors.append(self.get_query(i))
                _, _, query = self.inject_image_stack_vector(query, injected_vectors)
                        
            injected_key = self.get_key(1)  

            if injected_key is not None:
                key = self.get_key(0)
                value = self.get_value(0)
                    
                injected_keys, injected_values = [], []
                for i in range(1, len(self.qkv_feature_filenames)):
                    injected_keys.append(self.get_key(i))
                    injected_values.append(self.get_value(i))
                _, _, key = self.inject_image_stack_vector(key, injected_keys)
                _, _, value = self.inject_image_stack_vector(value, injected_values)

        hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=-1, is_causal=False
            )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype) # batch, pixels, channels

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        self.step += 1
        if self.step == self.num_inference_steps:
            self.finish()
        torch.cuda.empty_cache()

        return hidden_states