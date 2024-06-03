import os
import torch
import argparse
import torchvision

from pipeline_videogen import VideoGenPipeline
from pipelines.pipeline_inversion import VideoGenInversionPipeline 

from diffusers.schedulers import DDIMScheduler
from diffusers.models import AutoencoderKL
from diffusers.models import AutoencoderKLTemporalDecoder
from transformers import CLIPTokenizer, CLIPTextModel
from omegaconf import OmegaConf

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from utils import find_model
from models import get_models
import imageio
import decord
import numpy as np
from copy import deepcopy
from PIL import Image
from datasets import video_transforms
from torchvision import transforms

def prepare_image(path, vae, transform_video, device, dtype=torch.float16):
    with open(path, 'rb') as f:
        image = Image.open(f).convert('RGB')
    image = torch.as_tensor(np.array(image, dtype=np.uint8, copy=True)).unsqueeze(0).permute(0, 3, 1, 2)
    image, ori_h, ori_w, crops_coords_top, crops_coords_left = transform_video(image)
    image = vae.encode(image.to(dtype=dtype, device=device)).latent_dist.sample().mul_(vae.config.scaling_factor)
    image = image.unsqueeze(2)
    return image

def separation_content_motion(video_clip):
    """
    Separate content and motion in a given video.
    Args:
        video_clip: A given video clip, shape [B, C, F, H, W]

    Return:
        base_frame: Base frame, shape [B, C, 1, H, W]
        motions: Motions based on base frame, shape [B, C, F-1, H, W]
    """
    # Selecting the first frame from each video in the batch as the base frame
    base_frame = video_clip[:, :, :1, :, :]

    # Calculating the motion (difference between each frame and the base frame)
    motions = video_clip[:, :, 1:, :, :] - base_frame

    return base_frame, motions


class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        
    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sr={self.sr},'
                    f'num_threads={self.num_threads})')
        return repr_str


def main(args):
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 # torch.float16

    unet = get_models(args).to(device, dtype=torch.float16)
    state_dict = find_model(args.ckpt)
    unet.load_state_dict(state_dict)
    
    if args.enable_vae_temporal_decoder:
        if args.use_dct:
            vae_for_base_content = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_path, subfolder="vae_temporal_decoder", torch_dtype=torch.float64).to(device)
        else:
            vae_for_base_content = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_path, subfolder="vae_temporal_decoder", torch_dtype=torch.float16).to(device)
        vae = deepcopy(vae_for_base_content).to(dtype=dtype)
    else:
        vae_for_base_content = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae",).to(device, dtype=torch.float64)
        vae = deepcopy(vae_for_base_content).to(dtype=dtype)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)

    # set eval mode
    unet.eval()
    vae.eval()
    text_encoder.eval()

    scheduler_inversion = DDIMScheduler.from_pretrained(args.pretrained_model_path, 
                                              subfolder="scheduler",
                                              beta_start=args.beta_start, 
                                              beta_end=args.beta_end, 
                                              beta_schedule=args.beta_schedule,)
    
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, 
                                            subfolder="scheduler",
                                            beta_start=args.beta_start, 
                                            beta_end=args.beta_end, 
                                            # beta_end=0.017, 
                                            beta_schedule=args.beta_schedule,)

    videogen_pipeline = VideoGenPipeline(vae=vae, 
                                 text_encoder=text_encoder, 
                                 tokenizer=tokenizer, 
                                 scheduler=scheduler_inversion, 
                                 unet=unet).to(device)
    
    videogen_pipeline_inversion = VideoGenInversionPipeline(vae=vae, 
                                 text_encoder=text_encoder, 
                                 tokenizer=tokenizer, 
                                 scheduler=scheduler, 
                                 unet=unet).to(device)
    # videogen_pipeline.enable_xformers_memory_efficient_attention()
    # videogen_pipeline.enable_vae_slicing()

    transform_video = video_transforms.Compose([
        video_transforms.ToTensorVideo(),
        video_transforms.SDXLCenterCrop((args.image_size[0], args.image_size[1])), # center crop using shor edge, then resize
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])


    # video_path = './video_editing/A_man_walking_on_the_beach.mp4'
    video_path = './video_editing/a_corgi_walking_in_the_park_at_sunrise_oil_painting_style.mp4'


    video_reader = DecordInit()
    video = video_reader(video_path)
    frame_indice = np.linspace(0, 15, 16, dtype=int)
    video = torch.from_numpy(video.get_batch(frame_indice).asnumpy()).permute(0, 3, 1, 2).contiguous()
    video = video / 255.0
    video = video * 2.0 - 1.0
    latents = vae.encode(video.to(dtype=torch.float16, device=device)).latent_dist.sample().mul_(vae.config.scaling_factor).unsqueeze(0).permute(0, 2, 1, 3, 4)

    base_content, motion_latents = separation_content_motion(latents)

    # image_path = "./video_editing/a_man_walking_in_the_park.png"
    image_path = "./video_editing/a_cute_corgi_walking_in_the_park.png"
    edit_content = prepare_image(image_path, vae, transform_video, device, dtype=torch.float16).to(device)

    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path)

    # prompt_inversion = 'a man walking on the beach'
    prompt_inversion = 'a corgi walking in the park at sunrise, oil painting style'
    latents = videogen_pipeline_inversion(prompt_inversion, 
                                latents=motion_latents,
                                base_content=base_content,
                                video_length=args.video_length, 
                                height=args.image_size[0], 
                                width=args.image_size[1], 
                                num_inference_steps=args.num_sampling_steps,
                                guidance_scale=1.0,
                                # guidance_scale=args.guidance_scale,
                                motion_bucket_id=args.motion_bucket_id,
                                output_type="latent").video

    # prompt = 'a man walking in the park'
    prompt = 'a corgi walking in the park at sunrise, oil painting style'
    videos = videogen_pipeline(prompt, 
                               latents=latents,
                               base_content=edit_content,
                               video_length=args.video_length, 
                               height=args.image_size[0], 
                               width=args.image_size[1], 
                               num_inference_steps=args.num_sampling_steps,
                               guidance_scale=1.0,
                               #    guidance_scale=args.guidance_scale,
                               motion_bucket_id=args.motion_bucket_id,
                               enable_vae_temporal_decoder=args.enable_vae_temporal_decoder).video
    imageio.mimwrite(args.save_img_path + prompt.replace(' ', '_') + '_%04d' % args.run_time + '-imageio.mp4', videos[0], fps=8, quality=8) # highest quality is 10, lowest is 0
    print('save path {}'.format(args.save_img_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/sample.yaml")
    args = parser.parse_args()

    main(OmegaConf.load(args.config))


