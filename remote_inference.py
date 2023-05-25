import requests
import os
from PIL import Image
import base64
import io


def call_api(image_base64, text, min_length, max_length, beam_size, length_penalty, repetition_penalty, top_p,
             decoding_method):
    url = "https://4bd685a463b0b4c563.gradio.live/run/predict"
    payload = {
        "data": [
            image_base64,
            text,
            min_length,
            max_length,
            beam_size,
            length_penalty,
            repetition_penalty,
            top_p,
            decoding_method,
        ]
    }
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()["data"]
    else:
        return None


def process_images_in_folder(folder_path, text, min_length, max_length, beam_size, length_penalty, repetition_penalty,
                             top_p, decoding_method):
    # Iterate over files in the given directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".webp"):
            # Open the image file and convert it to base64
            with Image.open(os.path.join(folder_path, filename)) as img:
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")  # save as PNG format to ensure compatibility
                img_base64 = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

            # Call the API and print the result
            result = call_api(img_base64, text, min_length, max_length, beam_size, length_penalty, repetition_penalty,
                              top_p, decoding_method)
            print(f"Result for {filename}: {result}")



def actual_call_api():
    folder_path = r"D:\Andrew\Pictures\==Train\benchmark"
    text = "Can you tell me about this image in the most detailed and accurate way possible? In the form of general appearance, details, and then the image style, in a descriptive tone."
    min_length = 1
    max_length = 250
    beam_size = 5
    length_penalty = 1
    repetition_penalty = 1
    top_p = 0.9
    decoding_method = "Beam search"

    process_images_in_folder(folder_path, text, min_length, max_length, beam_size, length_penalty, repetition_penalty,
                             top_p, decoding_method)


if __name__ == '__main__':
    actual_call_api()

