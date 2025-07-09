import os
import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTConfig, SFTTrainer
from transformers.integrations import MLflowCallback
MLflowCallback.use_mlflow = False # to avoid logging parameters that may exceed Azure’s length limits.

# diable mlflow logingg
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"

# get the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

# Make dirs
os.makedirs(args.output_dir, exist_ok=True)
model_dir = os.path.join(args.output_dir, "model")
os.makedirs(model_dir, exist_ok=True)

# set few constants
max_seq_length = 1024

# Build full paths to the dataset files
train_file = os.path.join(args.dataset_path, "train.jsonl")
val_file = os.path.join(args.dataset_path, "dev.jsonl")
test_file = os.path.join(args.dataset_path, "test.jsonl")

# load the medqa usmle dataset
dataset = load_dataset("json", data_files={
    "train": train_file,
    "validation": val_file,
    "test": test_file
})
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

# load the model
model_name = "microsoft/Phi-4-mini-instruct"

# load the tokenizer first (needed for chat template)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.model_max_length = max_seq_length
tokenizer.padding_side = "right"

# convert to chat completion format
def formatting_func(example):
    question = example["question"]
    options = example["options"]
    answer_idx = example["answer_idx"]

    # Format options as A. Option text...
    formatted_options = "\n".join([f"{key}. {val}" for key, val in sorted(options.items())])
    
    user_prompt = f"Question:\n{question}\n\nOptions:\n{formatted_options}"

    # Create conversation in chat format
    messages = [
        {
            "role": "system",
            "content": "You are a medical expert. Read the following USMLE question and choose the best answer. Just give me the option letter (A, B, C, D, or E) as your answer."
        },
        {
            "role": "user",
            "content": user_prompt
        },
        {
            "role": "assistant",
            "content": answer_idx
        }
    ]
    
    # Apply chat template
    return tokenizer.apply_chat_template(messages, tokenize=False)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    use_cache=False,
    trust_remote_code=True,
    device_map="auto"
)

# lora config and apply lora
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# print lora trainable parameters
model.print_trainable_parameters()

# SFT config
training_config = SFTConfig(
    output_dir=args.output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    max_steps=-1,
    max_seq_length=max_seq_length,
    learning_rate=2e-5,
    log_level="info",
    logging_steps = 100,
    logging_strategy="steps",
    save_steps=500,
    save_strategy="steps",
    eval_strategy="epoch",
    seed=123,
    bf16=True
)

# SFT Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_config,
    processing_class=tokenizer,
    formatting_func=formatting_func,
    callbacks=[] # disables MLflow and other callbacks 
)

# train the model
trainer.train()

# save the model
print(f"Saving model to: {model_dir}")
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)