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
    def inference(image, prompts, num_captions, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p,
                  decoding_method,
                  modeltype):
        use_nucleus_sampling = decoding_method == "Nucleus sampling"
        print("model type: ", modeltype)

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

    num_captions = gr.Slider(
        minimum=1,
        maximum=50,
        value=1,
        step=1,
        interactive=True,
        label="num_captions",
    )

    min_len = gr.Slider(
        minimum=1,
        maximum=50,
        value=1,
        step=1,
        interactive=True,
        label="Min Length",
    )

    max_len = gr.Slider(
        minimum=10,
        maximum=500,
        value=250,
        step=5,
        interactive=True,
        label="Max Length",
    )

    sampling = gr.Radio(
        choices=["Beam search", "Nucleus sampling"],
        value="Beam search",
        label="Text Decoding Method",
        interactive=True,
    )

    top_p = gr.Slider(
        minimum=0.5,
        maximum=1.0,
        value=0.9,
        step=0.1,
        interactive=True,
        label="Top p",
    )

    beam_size = gr.Slider(
        minimum=1,
        maximum=10,
        value=5,
        step=1,
        interactive=True,
        label="Beam Size",
    )

    len_penalty = gr.Slider(
        minimum=-1,
        maximum=2,
        value=1,
        step=0.2,
        interactive=True,
        label="Length Penalty",
    )

    repetition_penalty = gr.Slider(
        minimum=-1,
        maximum=3,
        value=1,
        step=0.2,
        interactive=True,
        label="Repetition Penalty",
    )

    prompt_textbox = gr.Textbox(label="Prompt:", placeholder="prompt", lines=4)

    gr.Interface(
        fn=inference,
        inputs=[image_input, prompt_textbox, num_captions, min_len, max_len, beam_size, len_penalty, repetition_penalty,
                top_p,
                sampling],
        outputs="text",
        allow_flagging="never",
    ).launch(share=True, server_port=7861)


from fastapi import FastAPI
from multiprocessing import Process
import uvicorn

app = FastAPI()


@app.get('/')
def read_root():
    return {"message": "Hello, FastAPI"}


def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == '__main__':
    # start FastAPI in a separate process
    Process(target=run_fastapi).start()
    launch_gradio()
