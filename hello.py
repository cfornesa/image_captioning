# Import gradio as a UI wrapper
import gradio as gr

# Simple greeting function
def greet(name):
    return "Hello " + name + "!"

# Gradio interface wrapper for the greet function
    # Text input and text output
demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# Launch a server to serve the demo at port 7860
demo.launch(server_name="0.0.0.0", server_port=7860)