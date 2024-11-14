import streamlit as st
import ollama
import time
import asyncio
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont

# Load Stable Diffusion models
text2img_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
text2img_pipe = text2img_pipe.to("cuda")
img2img_pipe = img2img_pipe.to("cuda")


# Streamlit Sidebar with tabs
with st.sidebar:
    st.write("**Ollama + Stablediffusion**")
    st.write("[View the source code](https://github.com/ELIXREIX/NLP-Project)")

    # Sidebar tab for image modification
    tab_selection = st.radio("Select an option:", ["Chatbot", "Modify Image", "Generate Image"], index=0)

# App title
st.title("💬 Chatbot with Image Generation and Modification")

# Check if any keyword in the prompt matches the list of image keywords
def contains_image_keyword(prompt):
    return any(keyword in prompt.lower() for keyword in image_keywords)

# Initialize chat history and image state if not already in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "image_uploaded" not in st.session_state:
    st.session_state["image_uploaded"] = False  # Track if an image has been uploaded

image_keywords = [
    "generate photo", "show photo", "create photo", "make photo", "produce image",
    "generate image", "show image", "create image", "make image", "render image",
    "draw a picture", "paint a picture", "illustrate", "sketch an image", "artwork of",
    "art of", "create artwork", "make drawing", "visualize", "display photo of",
    "create scene of", "generate art of", "edit photo", "modify image", "improve image",
    "picture of", "image of", "snap a photo", "show me", "make a version of", "produce photo",
    "display picture", "generate a version of", "render photo", "make art"
]

# Chatbot Tab
if tab_selection == "Chatbot":
    # Display previous chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Typing effect function
    def type_text(placeholder, text, delay=0.005):
        typed_text = ""
        for char in text:
            typed_text += char
            placeholder.empty()  
            placeholder.write(typed_text)  
            time.sleep(delay)

    # Generate chatbot response using Ollama
    async def generate_response(prompt):
        full_response = ollama.generate("llama3.2", prompt)
        return full_response['response']


# Handle user input
if prompt := st.chat_input(key="unique_chat_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Check if the prompt includes any keyword for image generation
    if contains_image_keyword(prompt):
        with st.spinner("Generating image..."):
            # Use Stable Diffusion for image generation
            image = text2img_pipe(prompt).images[0]
            st.image(image, caption=f"Generated Image for: '{prompt}'", use_column_width=True)
            st.session_state.messages.append({"role": "assistant", "content": "[Image generated]"})
            st.session_state["image_uploaded"] = False  # Reset image upload state for next actions
    
    else:
        # Generate a text response if not an image request
        with st.spinner("Generating response..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_msg = loop.run_until_complete(generate_response(prompt))

            assistant_message_placeholder = st.empty()  
            type_text(assistant_message_placeholder, response_msg)

        st.session_state.messages.append({"role": "assistant", "content": response_msg})

# Generate Image Tab
if tab_selection == "Generate Image":
    st.write("**Enter a prompt to generate an image:**")
    
    # Input box for the user to enter prompt for image generation
    prompt = st.text_input("Enter image description", "")

    # Additional options for image generation
    st.write("**Advanced Image Generation Options**")

    # Image resolution options
    resolution = st.selectbox("Select Image Resolution", ["Low (256x256)", "Medium (512x512)", "High (1024x1024)"], index=1)

    # Creativity / randomness control
    creativity = st.slider("Creativity (Influences style variation)", 0.0, 1.0, 0.7, step=0.05)

    # Input for seed value (deterministic generation)
    seed = st.number_input("Enter a Seed Value (Optional)", min_value=0, max_value=2**32-1, value=42)

    # Option to generate multiple variations
    num_variations = st.slider("Number of Variations", 1, 5, 1, step=1)

    # Guidance scale to control adherence to prompt
    guidance_scale = st.slider("Guidance Scale", 7.5, 20.0, 12.0, step=0.5)

    if prompt:
        with st.spinner("Generating image..."):
            try:
                # Adjust resolution based on user selection
                if resolution == "Low (256x256)":
                    height, width = 256, 256
                elif resolution == "Medium (512x512)":
                    height, width = 512, 512
                else:
                    height, width = 1024, 1024

                # Generate image with specific options (creativity, seed, and guidance scale)
                generator = torch.manual_seed(seed) if seed else None
                image = text2img_pipe(prompt, height=height, width=width, guidance_scale=guidance_scale, generator=generator).images[0]

                # Display the generated image
                st.image(image, caption=f"Generated Image for: '{prompt}'", use_column_width=True)

                # Optionally generate multiple variations
                for i in range(1, num_variations):
                    variation_image = text2img_pipe(prompt, height=height, width=width, guidance_scale=guidance_scale, generator=generator).images[0]
                    st.image(variation_image, caption=f"Variation {i+1} for: '{prompt}'", use_column_width=True)

            except Exception as e:
                st.error(f"An error occurred while generating the image: {e}")

# Modify Image Tab
elif tab_selection == "Modify Image":
    st.write("**Upload an image to preview and modify:**")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"], key="image_uploader")

    if uploaded_file is not None:
        # Convert uploaded file to PIL Image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Preview Image", use_column_width=True)

        # Modify image options - displayed after image upload
        st.write("**Image Adjustments**")
        brightness = st.slider("Adjust Brightness", 0.5, 2.0, 1.0, step=0.1)
        contrast = st.slider("Adjust Contrast", 0.5, 2.0, 1.0, step=0.1)
        sharpness = st.slider("Adjust Sharpness", 0.5, 2.0, 1.0, step=0.1)
        blur = st.slider("Adjust Blur", 0, 10, 0, step=1)
        saturation = st.slider("Adjust Saturation", 0.5, 2.0, 1.0, step=0.1)

        # Rotation and flipping options
        rotate = st.slider("Rotate Image", 0, 360, 0, step=90)
        flip_option = st.radio("Flip Image", ["None", "Horizontal", "Vertical"])

        # Resize options
        resize = st.checkbox("Resize Image", value=False)
        if resize:
            width = st.slider("Width", 100, 1000, 512)
            height = st.slider("Height", 100, 1000, 512)

        # Grayscale option
        grayscale = st.checkbox("Convert to Grayscale", value=False)

        # Add text overlay option
        text_overlay = st.text_input("Text Overlay", "")
        text_color = st.color_picker("Text Color", "#000000")

        # Prompt for image modification (text input)
        modify_prompt = st.text_input("Modification Prompt", "Describe how to modify the image")

        # Button to confirm modification
        if st.button("Modify Image"):
            with st.spinner("Modifying image..."):
                # Apply brightness, contrast, sharpness, saturation, and blur adjustments
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness)
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast)
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(sharpness)
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(saturation)
                if blur > 0:
                    image = image.filter(ImageFilter.GaussianBlur(blur))

                # Apply rotation and flip
                if rotate != 0:
                    image = image.rotate(rotate)
                if flip_option == "Horizontal":
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                elif flip_option == "Vertical":
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)

                # Resize if the option is selected
                if resize:
                    image = image.resize((width, height))

                # Convert to grayscale if selected
                if grayscale:
                    image = ImageOps.grayscale(image)

                # Add text overlay if provided
                if text_overlay:
                    draw = ImageDraw.Draw(image)
                    font = ImageFont.load_default()
                    text_width, text_height = draw.textsize(text_overlay, font)
                    position = (image.width // 2 - text_width // 2, image.height // 2 - text_height // 2)
                    draw.text(position, text_overlay, fill=text_color, font=font)

                # Apply any text-based modification (if provided)
                if modify_prompt:
                    resized_image = image.resize((512, 512))
                    modified_image = img2img_pipe(prompt=modify_prompt, image=resized_image, strength=0.75).images[0]
                    image = modified_image

                # Display the modified image
                st.image(image, caption=f"Modified Image for: '{modify_prompt}'", use_column_width=True)
                st.session_state["image_uploaded"] = True
