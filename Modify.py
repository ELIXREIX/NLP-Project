import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline

# Initialize Stable Diffusion Pipeline (adjust your model path or model name as needed)
@st.cache_resource
def load_pipeline():
    # Make sure to replace 'path/to/your/model' with your model path if you're using a local model
    return StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda" if torch.cuda.is_available() else "cpu")

# Load the pipeline
pipe = load_pipeline()

# Streamlit App Title
st.title("Chatbot with Image Upload and Modification")

# Initialize chat history if not already in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous chat messages
for msg in st.session_state.messages:
    st.write(msg)

# Handle user input
if prompt := st.text_input("Ask a question or describe an image modification:"):
    st.session_state.messages.append(f"User: {prompt}")
    st.write(f"You asked: {prompt}")

# File uploader for images
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Button to generate modified image
    if st.button("Generate Modified Image"):
        with st.spinner("Generating modified image..."):
            # Run the pipeline and generate the new image
            result_image = pipe(prompt, init_image=image, strength=0.75).images[0]
            st.image(result_image, caption="Modified Image", use_column_width=True)

# Optionally, add chat history display
if st.button("Show Chat History"):
    st.write("Chat History:")
    for msg in st.session_state.messages:
        st.write(msg)
