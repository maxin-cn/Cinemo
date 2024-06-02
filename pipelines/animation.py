import os
import torch
import argparse
import torchvision

from pipeline_videogen import VideoGenPipeline
from diffusers.schedulers import DDIMScheduler
from diffusers.models import AutoencoderKL
from diffusers.models import AutoencoderKLTemporalDecoder
from transformers import CLIPTokenizer, CLIPTextModel
from omegaconf import OmegaConf

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from models import get_models
from utils import find_model
import imageio
from PIL import Image
import numpy as np
from datasets import video_transforms
from torchvision import transforms
import time
from einops import rearrange, repeat
from utils import dct_low_pass_filter, exchanged_mixed_dct_freq
from copy import deepcopy

def prepare_image(path, vae, transform_video, device, dtype=torch.float16):
    with open(path, 'rb') as f:
        image = Image.open(f).convert('RGB')
    image = torch.as_tensor(np.array(image, dtype=np.uint8, copy=True)).unsqueeze(0).permute(0, 3, 1, 2)
    image, ori_h, ori_w, crops_coords_top, crops_coords_left = transform_video(image)
    image = vae.encode(image.to(dtype=dtype, device=device)).latent_dist.sample().mul_(vae.config.scaling_factor)
    image = image.unsqueeze(2)
    return image

def main(args):

    if args.seed:
        torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 # torch.float16

    unet = get_models(args).to(device, dtype=dtype)

    t0 = time.time()
    state_dict = find_model(args.ckpt)
    print("Model download time: ", time.time() - t0)
    unet.load_state_dict(state_dict)
    

    if args.enable_vae_temporal_decoder:
        if args.use_dct:
            vae_for_base_content = AutoencoderKLTemporalDecoder.from_pretrained("/mnt/hwfile/gcc/maxin/work/pretrained/t2v_required_models/", subfolder="vae_temporal_decoder", torch_dtype=torch.float64).to(device)
        else:
            vae_for_base_content = AutoencoderKLTemporalDecoder.from_pretrained("/mnt/hwfile/gcc/maxin/work/pretrained/t2v_required_models/", subfolder="vae_temporal_decoder", torch_dtype=torch.float16).to(device)
        vae = deepcopy(vae_for_base_content).to(dtype=dtype)
    else:
        vae_for_base_content = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae",).to(device, dtype=torch.float64)
        vae = deepcopy(vae_for_base_content).to(dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder", torch_dtype=dtype).to(device) # huge

    # set eval mode
    unet.eval()
    vae.eval()
    text_encoder.eval()

    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, 
                                              subfolder="scheduler",
                                              beta_start=args.beta_start, 
                                              beta_end=args.beta_end, 
                                              beta_schedule=args.beta_schedule)


    videogen_pipeline = VideoGenPipeline(vae=vae, 
                                 text_encoder=text_encoder, 
                                 tokenizer=tokenizer, 
                                 scheduler=scheduler, 
                                 unet=unet).to(device)
    # videogen_pipeline.enable_xformers_memory_efficient_attention()
    # videogen_pipeline.enable_vae_slicing()

    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path)

    transform_video = video_transforms.Compose([
            video_transforms.ToTensorVideo(),
            video_transforms.SDXLCenterCrop((args.image_size[0], args.image_size[1])), # center crop using shor edge, then resize
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    for i, (image, prompt) in enumerate(args.image_prompts):
        if args.use_dct:
            base_content = prepare_image("./animated_images/" + image, vae_for_base_content, transform_video, device, dtype=torch.float64).to(device)
        else:
            base_content = prepare_image("./animated_images/" + image, vae_for_base_content, transform_video, device, dtype=torch.float16).to(device)

        if args.use_dct:
            # filter params
            print("Using DCT!")
            base_content_repeat = repeat(base_content, 'b c f h w -> b c (f r) h w', r=15).contiguous()

            # define filter
            freq_filter = dct_low_pass_filter(dct_coefficients=base_content,
                                                    percentage=0.23)
            
            noise = torch.randn(1, 4, 15, 40, 64).to(device)

            # add noise to base_content
            diffuse_timesteps = torch.full((1,),int(975))
            diffuse_timesteps = diffuse_timesteps.long()
            
            # 3d content
            base_content_noise = scheduler.add_noise(
                original_samples=base_content_repeat.to(device), 
                noise=noise, 
                timesteps=diffuse_timesteps.to(device))
            
            # 3d content
            latents = exchanged_mixed_dct_freq(noise=noise,
                        base_content=base_content_noise,
                        LPF_3d=freq_filter).to(dtype=torch.float16)
            
        base_content = base_content.to(dtype=torch.float16)

        videos = videogen_pipeline(prompt, 
                                   latents=latents if args.use_dct else None,
                                   base_content=base_content,
                                   video_length=args.video_length, 
                                   height=args.image_size[0], 
                                   width=args.image_size[1], 
                                   num_inference_steps=args.num_sampling_steps,
                                   guidance_scale=args.guidance_scale,
                                   motion_bucket_id=args.motion_bucket_id,
                                   enable_vae_temporal_decoder=args.enable_vae_temporal_decoder).video
        
        imageio.mimwrite(args.save_img_path + prompt.replace(' ', '_') + '_%04d' % i + '_%04d' % args.run_time + '-imageio.mp4', videos[0], fps=8, quality=8) # highest quality is 10, lowest is 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/sample.yaml")
    args = parser.parse_args()

    main(OmegaConf.load(args.config))
