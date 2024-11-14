import streamlit as st
import ollama
import time
import asyncio
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

# Load Stable Diffusion models
text2img_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
text2img_pipe = text2img_pipe.to("cuda")
img2img_pipe = img2img_pipe.to("cuda")


# Streamlit Sidebar with tabs
with st.sidebar:
    st.write("**Ollama LLaMA Chatbot**")
    st.write("[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)")
    st.write("[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)")

    # Sidebar tab for image modification
    tab_selection = st.radio("Select an option:", ["Chatbot", "Modify Image"], index=0)

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

# Modify Image Tab
elif tab_selection == "Modify Image":
    st.write("**Upload an image to preview and modify:**")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"], key="image_uploader")

    if uploaded_file is not None:
        # Convert uploaded file to a PIL Image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Preview Image", use_column_width=True)

        # Text input for modification prompt
        modify_prompt = st.text_input("Enter a description for modification (e.g., 'add a sunset glow'): ")

        # Button to confirm modification
        if st.button("Modify Image"):
            with st.spinner("Modifying image..."):
                # Apply modification to the uploaded image
                resized_image = image.resize((512, 512))
                modified_image = img2img_pipe(prompt=modify_prompt, image=resized_image, strength=0.75).images[0]
                st.image(modified_image, caption=f"Modified Image for: '{modify_prompt}'", use_column_width=True)
                st.session_state["image_uploaded"] = True  # Update the state to indicate an image has been uploaded
