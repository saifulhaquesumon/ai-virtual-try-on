import streamlit as st
from PIL import Image
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from diffusers import StableDiffusionInpaintPipeline
import numpy as np

# --- App Configuration ---
st.set_page_config(
    page_title="AI Virtual Try-On",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Model Loading ---
# Use Streamlit's caching to load models only once.
@st.cache_resource
def load_models():
    """Loads and caches the AI models."""
    # Segmentation Model (CLIPSeg)
    segmentation_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    segmentation_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    # Inpainting Model (Stable Diffusion)
    inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    # Move pipeline to GPU if available
    if torch.cuda.is_available():
        inpainting_pipe = inpainting_pipe.to("cuda")

    return segmentation_processor, segmentation_model, inpainting_pipe

st.title("👕 AI Virtual Try-On")
st.write("Upload your photo, choose a clothing region, and describe the item you want to try on!")

# Load the models and display a spinner while loading
with st.spinner("Loading AI models... This might take a few minutes on first run."):
    seg_processor, seg_model, inpaint_pipe = load_models()

# --- User Interface ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Your Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Your Uploaded Image", use_column_width=True)

with col2:
    if uploaded_image:
        st.subheader("2. Select Region & Describe Clothing")

        # Define segmentation prompts based on region selection
        region = st.radio(
            "Select clothing region:",
            ("Upper Region", "Lower Region"),
            horizontal=True,
        )
        if region == "Upper Region":
            seg_prompts = ["a shirt", "a t-shirt", "a top", "a jacket"]
            example_prompt = "a blue denim jacket"
        else: # Lower Region
            seg_prompts = ["pants", "trousers", "a skirt", "jeans"]
            example_prompt = "black leather pants"

        prompt = st.text_input("Enter a description of the clothing:", value=example_prompt)

        generate_button = st.button("✨ Generate Try-On!")

# --- Processing and Display ---
if 'generate_button' in locals() and generate_button and uploaded_image:
    with st.spinner("Processing... This can take up to a minute."):
        # --- Step 1: Generate Mask with CLIPSeg ---
        inputs = seg_processor(text=seg_prompts, images=[image] * len(seg_prompts), padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = seg_model(**inputs)
        
        # Combine masks from all prompts for the selected region
        preds = outputs.logits.unsqueeze(1)
        combined_heatmaps = torch.sigmoid(preds).mean(dim=0).squeeze() # Average heatmaps

        # Convert heatmap to binary mask
        threshold = 0.5
        mask = (combined_heatmaps > threshold).float()
        
        # Refine the mask (optional, but good for better results)
        # Convert to PIL for processing
        mask_pil = Image.fromarray((mask.numpy() * 255).astype(np.uint8))
        
        # Resize mask to match original image
        mask_resized = mask_pil.resize(image.size)


        # --- Step 2: Inpaint with Stable Diffusion ---
        # Move inputs to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Resize image for the inpainting model if it's too large
        original_width, original_height = image.size
        target_width, target_height = (512, 512)
        
        image_resized = image.resize((target_width, target_height))
        mask_for_inpaint = mask_resized.resize((target_width, target_height))

        # Perform inpainting
        result_image = inpaint_pipe(
            prompt=prompt,
            image=image_resized,
            mask_image=mask_for_inpaint,
            guidance_scale=7.5
        ).images[0]

        # Resize result back to original dimensions
        result_image = result_image.resize((original_width, original_height))

    # --- Display Results ---
    st.subheader("🎉 Your Virtual Try-On Result!")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.image(result_image, caption="Final Try-On Image", use_column_width=True)
    with res_col2:
        st.image(mask_resized, caption="Generated Mask", use_column_width=True)

else:
    st.info("Please upload an image and fill in the details to get started.")