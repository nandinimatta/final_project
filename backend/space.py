import sys
import subprocess
import os

# 1. Install gradio dynamically if not present (for Hugging Face Spaces environment)
try:
    import gradio as gr
except ImportError:
    print("[INFO] Gradio not found. Installing dynamically for Hugging Face Spaces...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
    import gradio as gr

import spaces
import uvicorn
import huggingface_hub

# 2. Monkeypatch HfFolder which was removed in modern huggingface_hub versions
class FakeHfFolder:
    @classmethod
    def get_token(cls):
        return None
    @classmethod
    def save_token(cls, token):
        pass
    @classmethod
    def delete_token(cls):
        pass
huggingface_hub.HfFolder = FakeHfFolder
sys.modules['huggingface_hub.HfFolder'] = FakeHfFolder
for mod_name in ['huggingface_hub', 'huggingface_hub.hf_api']:
    if mod_name in sys.modules:
        setattr(sys.modules[mod_name], 'HfFolder', FakeHfFolder)

# 3. Define the dummy GPU function at the top level for ZeroGPU scanning
@spaces.GPU
def dummy_gpu_trigger(name=""):
    return "ZeroGPU Triggered"

# 4. Define the Gradio greet function (also decorated to be safe)
@spaces.GPU
def greet(name):
    return "SoftPredict Clinical API Server is Running!"

# 5. Import the FastAPI app from app.py
from app import app

# 6. Create the Gradio interface
demo = gr.Interface(
    fn=greet, 
    inputs="text", 
    outputs="text", 
    title="SoftPredict Clinical API",
    description="The FastAPI backend is fully operational."
)

# 7. Mount the FastAPI app on the Gradio app instance
app = gr.mount_gradio_app(app, demo, path="/gradio")

# 8. Start the server directly to block and keep the space alive
if __name__ == "__main__":
    print("[INFO] Starting Uvicorn server on port 7860...")
    uvicorn.run(app, host="0.0.0.0", port=7860)
