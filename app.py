import os
import base64
import io
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
from PIL import Image

# --- PAGE SETUP ---
st.set_page_config(page_title="Pro Headshot Generator", page_icon="📸", layout="centered")

# --- MOBILE-FIRST STYLES ---
st.markdown("""
<style>
    .block-container { padding: 1.5rem 1rem; max-width: 480px; }
    .stButton > button { width: 100%; height: 3.2rem; font-size: 1.1rem; border-radius: 12px; }
    h1 { font-size: 1.6rem !important; }
    .step-label { font-size: 1rem; font-weight: 600; margin-bottom: 0.25rem; color: #aaa; }
    [data-testid="stSidebar"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- CAMERA COMPONENT ---
# Custom component: capture="user" is baked into the HTML — no JS injection,
# no WebRTC stream. Mobile browsers open the native camera at full resolution.
_COMPONENT_DIR = os.path.join(os.path.dirname(__file__), "camera_component")
_camera_capture = components.declare_component("camera_capture", path=_COMPONENT_DIR)

def camera_input(key):
    data_url = _camera_capture(key=key, default=None)
    if data_url:
        _, encoded = data_url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(encoded)))
    return None

# --- STATIC PROMPTS ---
NODE_01_HEADER = "Photorealistic, high-resolution, vertical, editorial head-and-chest portrait. Maintain accurate and realistic skin texture."
NODE_02_POSE = "The individual's body is slightly angled camera right (subject left), while their head is turned fully toward the camera. Their chin is slightly raised, shot from a low angle to create a sense of presence."
NODE_05_LIGHTING = "The lighting is dramatic, high-contrast chiaroscuro. A single pure, neutral white 5600k, small, gridded dish from camera left, casting a head focused, pool of light, creating high contrast and casting deep, sharp shadows; this light is strictly colorless and untinted. A strong, hard rim light with a vivid #E5E9A2 tint from camera right, sculpts their hair and shoulder, creating a sharp, defined edge. The background is a solid, artificial color block of faded-lime #E5E9A2, with a subtle gradient becoming brighter at the top, creating a clean, graphic, modern aesthetic. The vibe is Editorial mood, sharp focus, sculptural definition, contemporary style."
NODE_06_SYSTEM = "Accurately match image 1 wardrobe and facial features. No distortions, wrong gender presentation, mismatched skin tones, inaccurate facial features, double faces, warping, extra limbs, mutated hands, inconsistent lighting, incorrect background color, blurry textures, or cartoonish artifacts. No nudity. No visible lighting equipment. Editorial. Professional."
NODE_03_ANALYSIS_PROMPT = """Generate a simple 50 word description of the individual in the photo. Do not describe their surrounding, background, lighting, pose, or wardrobe. Only describe the person's features, facial accessories, whether they have an open or clothed mouth smile, and complexion. I want the format to be: "A (skin colour), (age estimation), (sex) with (hair colour and style). Their eye colour is (eye-colour). They have (facial hair description). They are wearing (accessories description). They have a (expression description)."""
NODE_04_WARDROBE_PROMPT = """Generate a simple, maximum of 35 word description of the wardrobe worn by the individual in the photo. Only add detail if necessary. I want the format to be: "The person is wearing a (wardrobe description)." """

# --- HELPER FUNCTIONS ---
def analyze_image(client, pil_image, prompt):
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[pil_image, prompt]
    )
    return response.text.strip()

def run_workflow(images, key):
    client = genai.Client(api_key=key)
    primary_image = images[0]

    with st.status("🕵️ Analyzing your photos...", expanded=True) as status:
        try:
            st.write("Extracting facial features...")
            face_desc = analyze_image(client, primary_image, NODE_03_ANALYSIS_PROMPT)
            st.write(f"Found: *{face_desc[:60]}...*")

            st.write("Extracting style & wardrobe...")
            wardrobe_desc = analyze_image(client, primary_image, NODE_04_WARDROBE_PROMPT)
            st.write(f"Found: *{wardrobe_desc[:60]}...*")

            status.update(label="Analysis Complete", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Analysis Failed", state="error", expanded=True)
            st.error(f"Analysis error: {e}")
            return None

    final_prompt = f"""
    {NODE_01_HEADER}
    {NODE_02_POSE}
    SUBJECT DESCRIPTION:
    {face_desc}
    {wardrobe_desc}
    {NODE_05_LIGHTING}
    """

    status_text = st.empty()
    status_text.text("🎨 Generating your headshot...")

    try:
        gen_contents = [final_prompt]
        gen_contents.extend(images)

        response = client.models.generate_content(
            model='gemini-3-pro-image-preview',
            contents=gen_contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                system_instruction=NODE_06_SYSTEM,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_ONLY_HIGH"
                    )
                ],
            )
        )

        status_text.empty()

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.inline_data:
                    return Image.open(io.BytesIO(part.inline_data.data))

        st.error("The model returned no image. It may have refused the request.")
        return None

    except Exception as e:
        st.error(f"Generation failed: {e}")
        return None

# --- UI ---
st.title("📸 Headshot Builder")
st.caption("Take 3 selfies to generate your editorial headshot.")

if "GOOGLE_API_KEY" not in st.secrets:
    st.error("No API key configured. Please add GOOGLE_API_KEY to your Streamlit secrets.")
    st.stop()

api_key = st.secrets["GOOGLE_API_KEY"]

# Sequential selfie capture using native high-res camera component
st.markdown('<div class="step-label">📷 Selfie 1 of 3</div>', unsafe_allow_html=True)
photo1 = camera_input(key="photo1")

photo2 = None
photo3 = None

if photo1:
    st.markdown('<div class="step-label">📷 Selfie 2 of 3</div>', unsafe_allow_html=True)
    photo2 = camera_input(key="photo2")

if photo2:
    st.markdown('<div class="step-label">📷 Selfie 3 of 3</div>', unsafe_allow_html=True)
    photo3 = camera_input(key="photo3")

if photo1 and photo2 and photo3:
    cols = st.columns(3)
    for i, img in enumerate([photo1, photo2, photo3]):
        cols[i].image(img, use_container_width=True)

    st.write("")
    if st.button("✨ Generate Headshot", type="primary"):
        result_image = run_workflow([photo1, photo2, photo3], api_key)
        if result_image:
            st.success("Generation Successful!")
            st.image(result_image, caption="Your Professional Headshot", use_container_width=True)
