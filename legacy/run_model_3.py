import gradio as gr
from lavis.models import load_model_and_preprocess
import torch
import argparse
import time

# =================
parser = argparse.ArgumentParser(description="Demo")
parser.add_argument("--model-name", default="blip2_vicuna_instruct")
parser.add_argument("--model-type", default="vicuna7b")
args = parser.parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

print('Loading model...')
start_time = time.time()

model, vis_processors, _ = load_model_and_preprocess(
    name=args.model_name,
    model_type=args.model_type,
    is_eval=True,
    device=device,
)
end_time = time.time()
print('Loading model done! Time cost: ', end_time - start_time, 's')


# ==================

def launch_gradio():
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
        prompts = prompts.split('\n')
        assert len(prompts) == 4, "The number of prompts must be equal to 4."

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

        # Now outputs is a list of generated texts for each prompt
        return '\n'.join(outputs)

    image_input = gr.Image(type="pil")

    prompt_textbox = gr.Textbox(label="Prompt:", placeholder="prompt", lines=4)

    num_captions = gr.Slider(
        minimum=1,
        maximum=50,
        value=1,
        step=1,
        interactive=True,
        label="num_captions",
    )

    gr.Interface(
        fn=inference,
        inputs=[image_input, prompt_textbox, num_captions],
        outputs="text",
        allow_flagging="never",
    ).launch(share=True, server_port=7861)


from fastapi import FastAPI
app = FastAPI()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-name", default="blip2_vicuna_instruct")
    parser.add_argument("--model-type", default="vicuna7b")
    args = parser.parse_args()
    launch_gradio()
