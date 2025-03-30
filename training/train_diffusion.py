from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
from accelerate import Accelerator

# Initialize accelerator
accelerator = Accelerator()

# Load pretrained Waifu Diffusion
model_id = "hakurei/waifu-diffusion"
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

# Prepare dataset (simplified)
dataset = ...  # Load your custom dataset
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)

# Accelerate setup
unet, optimizer, train_dataloader = accelerator.prepare(
    unet, optimizer, train_dataloader
)

# Training loop
for epoch in range(10):
    for batch in train_dataloader:
        # Forward pass
        with accelerator.accumulate(unet):
            loss = unet(batch["pixel_values"], batch["input_ids"]).loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
    torch.save(unet.state_dict(), f"./models/diffusion/unet_epoch_{epoch}.pt")
