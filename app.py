import streamlit as st
import ollama
import time
import asyncio

# Streamlit Sidebar
with st.sidebar:
    st.write("**Ollama LLaMA Chatbot**")
    st.write("[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)")
    st.write("[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)")

# App title
st.title("💬 Chatbot")

# Initialize chat history if not already in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display previous chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Function to display text character by character with typing effect
def type_text(placeholder, text, delay=0.005):
    typed_text = ""
    for char in text:
        typed_text += char
        placeholder.empty()  # Clear the placeholder
        placeholder.write(typed_text)  # Display typed text progressively
        time.sleep(delay)

# Function to generate the chatbot response using Ollama
async def generate_response(prompt):
    full_response = ollama.generate("llama3.2", prompt)  # Synchronous Ollama call
    return full_response['response']

# Handle user input (with a unique key)
if prompt := st.chat_input(key="unique_chat_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Display spinner while generating response
    with st.spinner("Generating response..."):
        loop = asyncio.new_event_loop()  # Create a new asyncio event loop
        asyncio.set_event_loop(loop)  # Set the event loop
        response_msg = loop.run_until_complete(generate_response(prompt))  # Wait for the response

        # Typing effect for the assistant's message
        assistant_message_placeholder = st.empty()  # Placeholder for typing effect
        type_text(assistant_message_placeholder, response_msg)

    # Append the assistant's complete response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_msg})
