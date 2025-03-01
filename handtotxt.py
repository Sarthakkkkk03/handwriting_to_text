import streamlit as st
import fitz  # pymupdf
import pdf2image
import base64
import time
import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from mistralai import Mistral


# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
model = "pixtral-12b-2409"
client = Mistral(api_key=MISTRAL_API_KEY)

# Function to convert PDF to base64 images using PyMuPDF
def pdf_to_base64_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)  # Open PDF
    base64_images = []

    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap()  # Convert page to image
        img_bytes = pix.tobytes("png")  # Convert to PNG format
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        base64_images.append(img_base64)

    return base64_images

# Streamlit UI
st.title("Handwritten PDF to Text Converter (PixTral + Mistral)")
st.write("Upload a PDF and extract handwritten text using PixTral-Mistral.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    try:
        # Save uploaded file temporarily
        pdf_path = "uploaded.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        st.write("Converting PDF to base64 images...")
        base64_images = pdf_to_base64_pymupdf(pdf_path)  # Using PyMuPDF function

        output_text_from_pdf = ""

        for idx, image in enumerate(base64_images):
            st.write(f"Processing Page {idx+1}...")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the Handwritten text in the given image"},
                        {"type": "image_url", "image_url": f"data:image/png;base64,{image}"}
                    ]
                }
            ]

            chat_response = client.chat.complete(
                model=model,
                messages=messages
            )

            extracted_text = chat_response.choices[0].message.content
            output_text_from_pdf += f"\nPage {idx+1}:\n{extracted_text}\n"
            
            st.subheader(f"Extracted Text from Page {idx+1}:")
            st.text(extracted_text)
            time.sleep(5)  # Avoid rate limits

    except Exception as e:
        st.error(f"An error occurred: {e}")
