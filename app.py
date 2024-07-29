import gradio as gr
import os
import torch
import argparse
import torchvision


from pipelines.pipeline_videogen import VideoGenPipeline
from diffusers.schedulers import DDIMScheduler
from diffusers.models import AutoencoderKL
from diffusers.models import AutoencoderKLTemporalDecoder
from transformers import CLIPTokenizer, CLIPTextModel
from omegaconf import OmegaConf

import os, sys
sys.path.append(os.path.split(sys.path[0])[0])
from models import get_models
import imageio
from PIL import Image
import numpy as np
from datasets import video_transforms
from torchvision import transforms
from einops import rearrange, repeat
from utils import dct_low_pass_filter, exchanged_mixed_dct_freq
from copy import deepcopy
import spaces
import requests
from datetime import datetime
import random
    
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./configs/sample.yaml")
args = parser.parse_args()
args = OmegaConf.load(args.config)

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 # torch.float16

unet = get_models(args).to(device, dtype=dtype)

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
text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder", torch_dtype=dtype).to(device) # huge

# set eval mode
unet.eval()
vae.eval()
text_encoder.eval()

basedir        = os.getcwd()
savedir        = os.path.join(basedir, "samples/Gradio", datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
savedir_sample = os.path.join(savedir, "sample")
os.makedirs(savedir, exist_ok=True)

def update_and_resize_image(input_image_path, height_slider, width_slider):
    if input_image_path.startswith("http://") or input_image_path.startswith("https://"):
        pil_image = Image.open(requests.get(input_image_path, stream=True).raw).convert('RGB')
    else:
        pil_image = Image.open(input_image_path).convert('RGB')
    
    original_width, original_height = pil_image.size

    if original_height == height_slider and original_width == width_slider:
        return gr.Image(value=np.array(pil_image))
    
    ratio1 = height_slider / original_height
    ratio2 = width_slider / original_width
    
    if ratio1 > ratio2:
        new_width = int(original_width * ratio1)
        new_height = int(original_height * ratio1)
    else:
        new_width = int(original_width * ratio2)
        new_height = int(original_height * ratio2)
    
    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    left = (new_width - width_slider) / 2
    top = (new_height - height_slider) / 2
    right = left + width_slider
    bottom = top + height_slider
    
    pil_image = pil_image.crop((left, top, right, bottom))
    
    return gr.Image(value=np.array(pil_image))


def update_textbox_and_save_image(input_image, height_slider, width_slider):
    pil_image = Image.fromarray(input_image.astype(np.uint8)).convert("RGB")

    original_width, original_height = pil_image.size
    
    ratio1 = height_slider / original_height
    ratio2 = width_slider / original_width
    
    if ratio1 > ratio2:
        new_width = int(original_width * ratio1)
        new_height = int(original_height * ratio1)
    else:
        new_width = int(original_width * ratio2)
        new_height = int(original_height * ratio2)
    
    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    left = (new_width - width_slider) / 2
    top = (new_height - height_slider) / 2
    right = left + width_slider
    bottom = top + height_slider
    
    pil_image = pil_image.crop((left, top, right, bottom))

    img_path = os.path.join(savedir, "input_image.png")
    pil_image.save(img_path)

    return gr.Textbox(value=img_path), gr.Image(value=np.array(pil_image))

def prepare_image(image, vae, transform_video, device, dtype=torch.float16):
    image = torch.as_tensor(np.array(image, dtype=np.uint8, copy=True)).unsqueeze(0).permute(0, 3, 1, 2)
    image = transform_video(image)
    image = vae.encode(image.to(dtype=dtype, device=device)).latent_dist.sample().mul_(vae.config.scaling_factor)
    image = image.unsqueeze(2)
    return image


@spaces.GPU
def gen_video(input_image, prompt, negative_prompt, diffusion_step, height, width, scfg_scale, use_dctinit, dct_coefficients, noise_level, motion_bucket_id, seed):

    torch.manual_seed(seed)

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

    transform_video = transforms.Compose([
        video_transforms.ToTensorVideo(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    if args.use_dct:
        base_content = prepare_image(input_image, vae_for_base_content, transform_video, device, dtype=torch.float64).to(device)
    else:
        base_content = prepare_image(input_image, vae_for_base_content, transform_video, device, dtype=torch.float16).to(device)

    if use_dctinit:
        # filter params
        print("Using DCT!")
        base_content_repeat = repeat(base_content, 'b c f h w -> b c (f r) h w', r=15).contiguous()

        # define filter
        freq_filter = dct_low_pass_filter(dct_coefficients=base_content, percentage=dct_coefficients)
        
        noise = torch.randn(1, 4, 15, 40, 64).to(device)

        # add noise to base_content
        diffuse_timesteps = torch.full((1,),int(noise_level))
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
                               negative_prompt=negative_prompt,
                               latents=latents if use_dctinit else None,
                               base_content=base_content,
                               video_length=15, 
                               height=height, 
                               width=width, 
                               num_inference_steps=diffusion_step,
                               guidance_scale=scfg_scale,
                               motion_bucket_id=100-motion_bucket_id,
                               enable_vae_temporal_decoder=args.enable_vae_temporal_decoder).video
    
    save_path = args.save_img_path + 'temp' + '.mp4'
    # torchvision.io.write_video(save_path, videos[0], fps=8, video_codec='h264', options={'crf': '10'})
    imageio.mimwrite(save_path, videos[0], fps=8, quality=7)
    return save_path


if not os.path.exists(args.save_img_path):
    os.makedirs(args.save_img_path)


with gr.Blocks() as demo:

    gr.Markdown("<font color=red size=6.5><center>Cinemo: Consistent and Controllable Image Animation with Motion Diffusion Models</center></font>")
    gr.Markdown(
        """<div style="display: flex;align-items: center;justify-content: center">
        [<a href="https://arxiv.org/abs/2407.15642">Arxiv Report</a>] | [<a href="https://https://maxin-cn.github.io/cinemo_project/">Project Page</a>] | [<a href="https://github.com/maxin-cn/Cinemo">Github</a>]</div>
        """
    )


    with gr.Column(variant="panel"):
        with gr.Row():
            prompt_textbox = gr.Textbox(label="Prompt", lines=1)
            negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=1)
            
        with gr.Row(equal_height=False):
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(label="Input Image", interactive=True)
                    result_video = gr.Video(label="Generated Animation", interactive=False, autoplay=True)
        
        generate_button = gr.Button(value="Generate", variant='primary')
        
        with gr.Accordion("Advanced options", open=False):
            gr.Markdown(
            """
            - Input image can be specified using the "Input Image URL" text box or uploaded by clicking or dragging the image to the "Input Image" box.
            - Input image will be resized and/or center cropped to a given resolution (320 x 512) automatically.
            - After setting the input image path, press the "Preview" button to visualize the resized input image.
            """
            )
            with gr.Column():
                with gr.Row():
                    input_image_path = gr.Textbox(label="Input Image URL", lines=1, scale=10, info="Press Enter or the Preview button to confirm the input image.")
                    preview_button = gr.Button(value="Preview")
                    
                with gr.Row():
                    sample_step_slider = gr.Slider(label="Sampling steps", value=50, minimum=10, maximum=250, step=1)

                with gr.Row():
                    seed_textbox = gr.Slider(label="Seed", value=100, minimum=1, maximum=int(1e8), step=1, interactive=True)
                    # seed_textbox = gr.Textbox(label="Seed", value=100)
                    # seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                    # seed_button.click(fn=lambda: gr.Textbox(value=random.randint(1, int(1e8))), inputs=[], outputs=[seed_textbox])
                
                with gr.Row():
                    height = gr.Slider(label="Height", value=320, minimum=0, maximum=512, step=16, interactive=False)
                    width  = gr.Slider(label="Width",  value=512, minimum=0, maximum=512, step=16, interactive=False)
                with gr.Row():
                    txt_cfg_scale = gr.Slider(label="CFG Scale",   value=7.5, minimum=1.0,   maximum=20.0, step=0.1, interactive=True)
                    motion_bucket_id = gr.Slider(label="Motion Intensity",   value=10, minimum=1,   maximum=20, step=1, interactive=True)
                
                with gr.Row():
                    use_dctinit = gr.Checkbox(label="Enable DCTInit", value=True)
                    dct_coefficients = gr.Slider(label="DCT Coefficients", value=0.23, minimum=0, maximum=1, step=0.01, interactive=True)
                    noise_level = gr.Slider(label="Noise Level", value=985, minimum=1, maximum=999, step=1, interactive=True)
        
        input_image.upload(fn=update_textbox_and_save_image, inputs=[input_image, height, width], outputs=[input_image_path, input_image])    
        preview_button.click(fn=update_and_resize_image, inputs=[input_image_path, height, width], outputs=[input_image])
        input_image_path.submit(fn=update_and_resize_image, inputs=[input_image_path, height, width], outputs=[input_image])

        EXAMPLES = [
            ["./example/aircrafts_flying/0.jpg", "aircrafts flying"                   , "", 50, 320, 512, 7.5, True, 0.23, 975, 10, 100],
            ["./example/fireworks/0.jpg", "fireworks"                                 , "", 50, 320, 512, 7.5, True, 0.23, 975, 10, 100],
            ["./example/flowers_swaying/0.jpg", "flowers swaying"                     , "", 50, 320, 512, 7.5, True, 0.23, 975, 10, 100],
            ["./example/girl_walking_on_the_beach/0.jpg", "girl walking on the beach" , "", 50, 320, 512, 7.5, True, 0.23, 985, 10, 200],
            ["./example/house_rotating/0.jpg", "house rotating"                       , "", 50, 320, 512, 7.5, True, 0.23, 985, 10, 100],
            ["./example/people_runing/0.jpg", "people runing"                         , "", 50, 320, 512, 7.5, True, 0.23, 975, 10, 100],
]

        examples = gr.Examples(
            examples = EXAMPLES,
            fn = gen_video,
            inputs=[input_image, prompt_textbox, negative_prompt_textbox, sample_step_slider, height, width, txt_cfg_scale, use_dctinit, dct_coefficients, noise_level, motion_bucket_id, seed_textbox],
            outputs=[result_video],
            # cache_examples=True,
            cache_examples="lazy",
        )

        generate_button.click(
                fn=gen_video,
                inputs=[
                    input_image,
                    prompt_textbox,
                    negative_prompt_textbox,
                    sample_step_slider,
                    height,
                    width,
                    txt_cfg_scale,
                    use_dctinit,
                    dct_coefficients,
                    noise_level,
                    motion_bucket_id,
                    seed_textbox,
                ],
                outputs=[result_video]
            )
    
demo.launch(debug=False, share=True, server_name="Cinemo")