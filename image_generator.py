from diffusers import StableDiffusionPipeline
import torch
import streamlit as st

# Load the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Use your GPU for image generation

# Streamlit App Title
st.title("🖼️ Image Generator")

# User Input for Image Prompt
prompt = st.text_input("Enter a prompt to generate an image")

# Generate Image Button
if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            image = pipe(prompt).images[0]
            st.image(image, caption=f"Generated Image for: '{prompt}'")
    else:
        st.warning("Please enter a prompt to generate an image.")
