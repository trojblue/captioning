import timeit
from tqdm.auto import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

# Load model and tokenizer
print("Loading model...")
model_start = timeit.default_timer()
model = LlamaForCausalLM.from_pretrained("/home/ubuntu/dev/vicuna-7B-1.1-HF")
model_end = timeit.default_timer()
print(f"Model loaded in: {model_end - model_start} seconds")

print("Loading tokenizer...")
tokenizer_start = timeit.default_timer()
tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/dev/vicuna-7B-1.1-HF")
tokenizer_end = timeit.default_timer()
print(f"Tokenizer loaded in: {tokenizer_end - tokenizer_start} seconds")

# If the tokenizer does not have a padding token, set it to the EOS token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Read prompts from file
with open('questions.txt', 'r') as file:
    prompts = file.read().splitlines()

print(f"Loaded {len(prompts)} prompts")

# Set batch size
batch_size = 10

# Iterate over prompts with progress tracking
for i in tqdm(range(0, len(prompts), batch_size), desc='Processing prompts'):
    batch_prompts = prompts[i:i + batch_size]

    # Convert all prompts to model inputs
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)

    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)

    # Decode all generated outputs
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # Print outputs
    for i, output in enumerate(outputs):
        print(f"Prompt: {batch_prompts[i]}")
        print(f"Response: {output}")
        print()
