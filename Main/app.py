import streamlit as st
from pathlib import Path
from PIL import Image

# LlamaIndex core
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Gemini LLM
from llama_index.llms.gemini import Gemini

# Optional: image captioning + OCR + audio transcription
from transformers import BlipProcessor, BlipForConditionalGeneration
import pytesseract
import whisper

# ------------------------------------------------------------
# Configure Gemini LLM + embeddings
# ------------------------------------------------------------
Settings.llm = Gemini(model="gemini-1.5-flash") 
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------------------------------------------
# Load index (from ingest.py)
# ------------------------------------------------------------
@st.cache_resource
def load_index(storage_dir="storage"):
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context)
    return index

index = load_index()
query_engine = index.as_query_engine()
llm = Settings.llm

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title("🤖 Multimodal Support Assistant (Gemini-powered)")

tab1, tab2, tab3 = st.tabs(["💬 Text", "🎤 Audio", "🖼️ Image"])

# ---- Text input
with tab1:
    query = st.text_input("Ask me anything about the PDF:")
    if query:
        response = query_engine.query(query)
        st.write("### Answer:")
        st.write(str(response))

# ---- Audio input
import tempfile

# ---- Audio input
with tab2:
    uploaded_audio = st.file_uploader("Upload audio file (wav/mp3)", type=["wav", "mp3"])
    if uploaded_audio:
        st.audio(uploaded_audio)

        # Save the uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_audio.read())
            tmp_audio_path = tmp_file.name

        # Load Whisper model
        model = whisper.load_model("base")
        transcription = model.transcribe(tmp_audio_path)["text"]

        st.write("**Transcribed text:**", transcription)

        response = query_engine.query(transcription)
        st.write("### Answer:")
        st.write(str(response))


# ---- Image input
with tab3:
    uploaded_img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    user_query = st.text_input("Enter your query related to the PDF + image")

    if uploaded_img and user_query:
        # Display image
        image = Image.open(uploaded_img)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # OCR extraction
        extracted_text = pytesseract.image_to_string(image)
        st.subheader("Extracted Text (OCR):")
        st.write(extracted_text if extracted_text.strip() else "⚠️ No text detected in image")

        # Optional: BLIP captioning
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            inputs = processor(image, return_tensors="pt")
            out = blip_model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            st.subheader("BLIP Caption:")
            st.write(caption)
        except Exception as e:
            caption = ""
            st.warning(f"BLIP captioning skipped: {e}")

        # Query the PDF
        pdf_response = query_engine.query(user_query)

        # Combine all sources
        final_prompt = f"""
        You are given:
        1. User query: {user_query}
        2. Extracted text from image (OCR): {extracted_text}
        3. Image caption (if any): {caption}
        4. Retrieved context from PDF: {pdf_response.response}

        Please generate a helpful answer combining both the PDF and image.
        """

        response = llm.complete(final_prompt)

        st.subheader("### Answer:")
        st.write(response.text)
