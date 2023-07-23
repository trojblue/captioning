import os
import io
import json
import base64
import requests
from PIL import Image
from tqdm.auto import tqdm

class GradioAPI:
    def __init__(self, url="https://aaa9600d4bc0a97c93.gradio.live/run/predict", text="", min_length=1, max_length=250,
                 beam_size=5, length_penalty=1, repetition_penalty=1, top_p=0.9, decoding_method="Beam search"):
        self.url = url
        self.text = text
        self.min_length = min_length
        self.max_length = max_length
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.decoding_method = decoding_method

    def call_api(self, image_base64):
        payload = {
            "data": [
                image_base64,
                self.text,
                self.min_length,
                self.max_length,
                self.beam_size,
                self.length_penalty,
                self.repetition_penalty,
                self.top_p,
                self.decoding_method,
            ]
        }
        response = requests.post(self.url, json=payload)
        return response.json()["data"] if response.status_code == 200 else None

    @staticmethod
    def resize_image(image, max_size=512):
        if max(image.size) > max_size:
            ratio = max_size / min(image.size)
            new_size = tuple(round(x*ratio) for x in image.size)
            image = image.resize(new_size, Image.ANTIALIAS)
        return image

    def convert_image_to_base64(self, image_path):
        with Image.open(image_path) as img:
            img = self.resize_image(img)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    def process_images_in_folder(self, folder_path):
        files = os.listdir(folder_path)
        for filename in tqdm(files, desc="Processing images"):
            if filename.endswith((".jpg", ".png", ".webp")):
                image_path = os.path.join(folder_path, filename)
                img_base64 = self.convert_image_to_base64(image_path)
                result = self.call_api(img_base64)
                self.save_to_json(result, folder_path, filename)

    def save_to_json(self, result, folder_path, filename):
        # Saving the response and the API parameters to a JSON file
        json_filename = os.path.splitext(filename)[0] + '.json'
        json_filepath = os.path.join(folder_path, json_filename)
        data = {
            'result': result,
            'parameters': {
                'text': self.text,
                'min_length': self.min_length,
                'max_length': self.max_length,
                'beam_size': self.beam_size,
                'length_penalty': self.length_penalty,
                'repetition_penalty': self.repetition_penalty,
                'top_p': self.top_p,
                'decoding_method': self.decoding_method
            }
        }
        with open(json_filepath, 'w') as f:
            json.dump(data, f)


def main():
    folder_path = r"D:\Andrew\Pictures\==Train\benchmark"
    text = "Question: what's in the image? answer:"
    URL = "https://fce0094b811d61e4be.gradio.live/run/predict"
    gradio = GradioAPI(text=text, url=URL)
    gradio.process_images_in_folder(folder_path)


if __name__ == '__main__':
    main()