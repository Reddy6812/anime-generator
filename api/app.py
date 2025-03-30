# anime-generator/api/app.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline

# Initialize FastAPI
app = FastAPI()

# Load your trained model
pipe = StableDiffusionPipeline.from_pretrained("./models/diffusion/")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_image(request: PromptRequest):
    # Generate image from prompt
    image = pipe(request.prompt).images[0]
    
    # Convert image to bytes (simplified example)
    from io import BytesIO
    import base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return JSONResponse(content={"image": img_str})
