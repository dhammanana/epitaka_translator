import os
from google import genai
from google.genai import types
from PIL import Image
import io

# 1. Setup Client (Ensure your API key is in your environment variables)
# Or use: client = genai.Client(api_key="YOUR_API_KEY")
client = genai.Client(api_key=os.environ.get("GEMINI_KEY_3"))

def generate_jataka_sample():
    print("Requesting image generation...")
    
    # 2. Call the model
    # Note: gemini-3.1-flash-image-preview is the standard 2026 free-tier image model
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-image",
        contents="A traditional Jataka tale illustration: a golden deer in a lush ancient Indian forest, intricate artistic style, vibrant colors.",
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        ),
    )

    # 3. Process and save the result
    for part in response.candidates.content.parts:
        if part.inline_data is not None:
            image_bytes = part.inline_data.data
            image = Image.open(io.BytesIO(image_bytes))
            image.save("jataka_sample.png")
            print("Success! Image saved as 'jataka_sample.png'.")
            image.show()
            return
            
    print("No image was returned in the response.")

if __name__ == "__main__":
    generate_jataka_sample()
