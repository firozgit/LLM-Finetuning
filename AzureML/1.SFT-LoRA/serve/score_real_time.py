import os
import sys
import traceback
import re
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Set up logging for Azure ML - this will appear in job logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # This ensures logs go to stdout/stderr
        logging.FileHandler('scoring.log')  # Also save to file
    ]
)
logger = logging.getLogger(__name__)

# Also print to stdout for Azure ML visibility
def print_and_log(message, level="INFO"):
    """Print to stdout and log - ensures visibility in Azure ML"""
    print(f"[{level}] {message}", flush=True)
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)

def init():
    """
    Initialize the model and tokenizer for batch inference.
    This function is called once when the scoring script starts.
    """
    global model, tokenizer, device
    
    try:
        print_and_log("Initializing LoRA model for batch inference...")
        
        # Get model path from environment variable
        model_dir = os.path.join(os.getenv("AZUREML_MODEL_DIR", "./"), "model")
        print_and_log(f"Loading model from: {model_dir}")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_and_log(f"Using device: {device}")
        
        # Load LoRA config
        print_and_log("Loading LoRA configuration...")
        
        peft_config = PeftConfig.from_pretrained(model_dir)
        print_and_log(f"Base model: {peft_config.base_model_name_or_path}")        
        
        # Load base model
        print_and_log("Loading base model...")        
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Load LoRA weights and merge
        print_and_log("Loading and merging LoRA weights...")        
        model = PeftModel.from_pretrained(base_model, model_dir)
        model.eval()
        
        # Load tokenizer
        print_and_log("Loading tokenizer...")        
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print_and_log("Set pad_token to eos_token")            
        
        print_and_log("Model initialization completed successfully!")        
        
    except Exception as e:
        print_and_log(f"Error during initialization: {str(e)}", "ERROR")                
        error_details = traceback.format_exc()
        print_and_log(f"Full traceback: {error_details}", "ERROR")        
        raise e
 
def run(raw_data):
    """
    Handle real-time inference (single request).
    """
    global model, tokenizer, device
    
    try:
        inputs = json.loads(raw_data)
        
        # Extract messages from the input data
        if "data" in inputs:
            messages = inputs["data"]
        elif "messages" in inputs:
            messages = inputs["messages"]
        else:
            # Fallback to treating the entire input as a prompt
            prompt = inputs.get("prompt", str(inputs))
            messages = [{"role": "user", "content": prompt}]
        
        # Get generation parameters from input or use defaults
        max_new_tokens = inputs.get("max_new_tokens", 10)
        temperature = inputs.get("temperature", 0.0)
        do_sample = inputs.get("do_sample", False)
        
        # Apply chat template to format the conversation
        if isinstance(messages, list):
            try:
                # If messages is a list of message objects, apply chat template
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                # Fallback to manual formatting
                formatted_prompt = format_chat_messages_manual(messages)
        else:
            # If it's already a string, use it directly
            formatted_prompt = messages
        
        # Tokenize the input
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (response)
        response_ids = output_ids[0][len(input_ids[0]):]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return response
        
    except Exception as e:
        print_and_log(f"Error in real-time inference: {str(e)}", "ERROR")
        return f"Error: {str(e)}"

def format_chat_messages_manual(messages: List[Dict]) -> str:
    """
    Manually format chat messages when tokenizer.apply_chat_template fails.
    """
    formatted_prompt = ""
    
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "system":
            formatted_prompt += f"System: {content}\n\n"
        elif role == "user":
            formatted_prompt += f"Human: {content}\n\n"
        elif role == "assistant":
            formatted_prompt += f"Assistant: {content}\n\n"
    
    # Add assistant prompt for generation
    formatted_prompt += "Assistant:"
    
    return formatted_prompt

def extract_answer(response: str) -> str:
    """
    Extract the answer choice (A, B, C, D, E) from the model response.
    """
    response = response.strip().upper()
    
    # Look for patterns like "A", "A.", "A)", "(A)", "The answer is A", etc.    
    
    # Pattern 1: Single letter at the start
    if len(response) >= 1 and response[0] in 'ABCDE':
        return response[0]
    
    # Pattern 2: Letter followed by punctuation
    match = re.search(r'([ABCDE])[\.\)\:]', response)
    if match:
        return match.group(1)
    
    # Pattern 3: "The answer is X" pattern
    match = re.search(r'(?:THE\s+)?ANSWER\s+IS\s+([ABCDE])', response)
    if match:
        return match.group(1)
    
    # Pattern 4: "(X)" pattern
    match = re.search(r'\(([ABCDE])\)', response)
    if match:
        return match.group(1)
    
    # Pattern 5: Find any single letter A-E in the response
    letters = re.findall(r'[ABCDE]', response)
    if letters:
        return letters[0]
    
    # If no clear answer found, return the first character if it's valid
    if response and response[0] in 'ABCDE':
        return response[0]
    
    # Default fallback
    print_and_log(f"Could not extract clear answer from: '{response}', defaulting to 'A'", "WARNING")
    return 'A'

# Comprehensive testing functions for local development
def test_single_request():
    """
    Test single request (real-time inference mode).
    """
    print("=" * 60)
    print("TESTING SINGLE REQUEST (Real-time Mode)")
    print("=" * 60)
    
    # Sample single request data
    single_request = {
        "messages": [
            {
                "role": "system",
                "content": "You are a medical expert. Read the following USMLE question and choose the best answer. Give me the answer as A/B/C/D/E."
            },
            {
                "role": "user",
                "content": "Question:\nWhat is the most common cause of pneumonia in healthy adults?\n\nOptions:\nA. Streptococcus pneumoniae\nB. Haemophilus influenzae\nC. Mycoplasma pneumoniae\nD. Legionella pneumophila\nE. Chlamydia pneumoniae"
            }
        ],
        "max_new_tokens": 10,
        "temperature": 0.0,
        "do_sample": False
    }
    
    try:
        # Convert to JSON string (as it would come from real-time endpoint)
        request_json = json.dumps(single_request)
        
        print(f"Input: {request_json[:100]}...")

        # Test single request using run directly
        response = run(request_json)
        
        print(f"\nSingle Request Results:")
        print(f"Response: {response}")
        print(f"Response Type: {type(response)}")
        
        # Extract answer from response
        if isinstance(response, str):
            extracted_answer = extract_answer(response)
            print(f"Extracted Answer: {extracted_answer}")
        
    except Exception as e:
        print(f"Error in single request test: {e}")        
        traceback.print_exc()

if __name__ == "__main__":
    """
    Main block for local testing and development.
    This will run when you execute the script directly: python score.py
    """   
    try:
        print("Initializing model for testing...")
        init()
        print("Model initialized successfully!\n")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        print("Make sure your model files are available in the correct location.")
        sys.exit(1)

    # Run single request test
    test_single_request()