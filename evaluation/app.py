import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("./models/diffusion/")
pipe = pipe.to(device)

def generate_anime(prompt):
    image = pipe(prompt, num_inference_steps=50).images[0]
    return image

gr.Interface(
    fn=generate_anime,
    inputs=gr.Textbox(label="Enter prompt (e.g., '1girl, green hair, smiling')"),
    outputs="image",
    title="Anime Generator"
).launch(share=True)
