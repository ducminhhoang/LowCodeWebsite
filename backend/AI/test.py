from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cpu" # the device to load the model onto
model_path = "01-ai/Yi-Coder-9B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").eval()

prompt = "Write a quick sort algorithm."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=1024,
    eos_token_id=tokenizer.eos_token_id  
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# device =  'cuda' if torch.cuda.is_available() else 'cpu' # the device to load the model onto
# model_path = "01-ai/Yi-Coder-9B-Chat"

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# # Initialize the model without loading the weights
# with init_empty_weights():
#     model = AutoModelForCausalLM.from_pretrained(model_path)

# # Offload to disk (provide an offload folder)
# model = load_checkpoint_and_dispatch(
#     model, model_path, device_map="auto", offload_folder="offload", offload_state_dict=True
# )

# model.eval()

# prompt = "Generate a complete HTML file for a simple login form with CSS styling and JavaScript validation."
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(device)

# generated_ids = model.generate(
#     model_inputs.input_ids,
#     max_new_tokens=1024,
#     eos_token_id=tokenizer.eos_token_id  
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)
