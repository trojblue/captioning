from lavis.models import load_model_and_preprocess
import torch
import argparse
import time

from PIL import Image

def inference(image, prompts, num_captions):
    min_len = 15
    max_len = 95
    beam_size = 5
    len_penalty = 1
    repetition_penalty = 1.2
    top_p = 0.9
    decoding_method = "Beam search"

    use_nucleus_sampling = decoding_method == "Nucleus sampling"

    # Ensure prompts are a list of strings
    # prompts = prompts.split('\n')
    # assert len(prompts) == 4, "The number of prompts must be equal to 4."

    image = vis_processors["eval"](image).unsqueeze(0).to(device)
    images = image.repeat(len(prompts), 1, 1, 1)

    samples = {
        "image": images,
        "prompt": prompts,
    }

    outputs = model.generate(
        samples,
        length_penalty=float(len_penalty),
        repetition_penalty=float(repetition_penalty),
        num_beams=beam_size,
        max_length=max_len,
        min_length=min_len,
        top_p=top_p,
        use_nucleus_sampling=use_nucleus_sampling,
        num_captions=num_captions
    )

    return outputs


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    print('Loading model...')
    start_time = time.time()

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_vicuna_instruct",
        model_type="vicuna7b",
        is_eval=True,
        device=device,
    )
    end_time = time.time()
    print('Loading model done! Time cost: ', end_time - start_time, 's')

    image_input = Image.open("test.jpg")
    print(type(image_input))
    print(image_input.size)

    prompts = ["what's in the image?", "Question: what's the image about? answer:"]
    samples = {
        "image": image_input,
        "prompt": prompts,
    }

    model.generate(samples)