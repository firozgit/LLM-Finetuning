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

model = None
tokenizer = None
device = None

# Also print to stdout for Azure ML visibility
import sys
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

def run(mini_batch):
    """
    Process a mini-batch of file paths for batch inference.
    
    Args:
        mini_batch: List of file paths to JSONL files
        
    Returns:
        List of processed file paths
    """
    global model, tokenizer, device
    
    print_and_log(f"=== RUN FUNCTION CALLED ===", "INFO")
    print_and_log(f"Input type: {type(mini_batch)}", "INFO")
    print_and_log(f"Input content preview: {str(mini_batch)[:300]}...", "INFO")
        
    # Process each file path in the mini-batch
    for file_path in mini_batch:
        print_and_log(f"Processing file: {file_path}", "INFO")
        
        # Read and process the JSONL file
        results = []
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse the JSON from each line
                    request = json.loads(line)
                    print_and_log(f"Processing request {line_num}: {request.get('id', f'line_{line_num}')}", "INFO")
                    
                    # Extract request parameters
                    messages = request.get("messages", [])
                    max_tokens = request.get("max_tokens", 10)
                    temperature = request.get("temperature", 0.0)
                    item_id = request.get("id", f"line_{line_num}")
                    ground_truth = request.get("ground_truth", "")
                    
                    # Handle different input formats
                    if "data" in request:
                        messages = request["data"]
                    elif "prompt" in request:
                        messages = [{"role": "user", "content": request["prompt"]}]
                    
                    # Ensure we have valid messages
                    if not messages or not isinstance(messages, list):
                        print_and_log(f"Invalid messages for {item_id}, using fallback", "WARNING")
                        messages = [{"role": "user", "content": str(request)}]
                    
                    # Format the prompt using chat template
                    try:
                        formatted_prompt = tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                    except Exception as e:
                        print_and_log(f"Chat template failed: {e}, using manual formatting", "WARNING")
                        formatted_prompt = format_chat_messages_manual(messages)
                    
                    # Tokenize the input
                    inputs = tokenizer(
                        formatted_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024,
                        padding=False
                    ).to(device)
                    
                    # Generate response
                    with torch.no_grad():
                        output_ids = model.generate(
                            inputs.input_ids,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            do_sample=temperature > 0,
                            top_p=0.9 if temperature > 0 else None,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    
                    # Decode the response
                    input_length = inputs.input_ids.shape[1]
                    generated_tokens = output_ids[0][input_length:]
                    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    
                    # Extract answer
                    answer = extract_answer(response)
                    
                    # Create result
                    result = {
                        "id": item_id,
                        "prediction": answer,
                        "full_response": response,
                        "ground_truth": ground_truth,
                        "prompt_length": input_length,
                        "response_length": len(generated_tokens)
                    }
                    
                    results.append(result)
                    print_and_log(f"Item {item_id}: Predicted '{answer}', Ground truth '{ground_truth}'", "INFO")
                    
                except Exception as e:
                    print_and_log(f"Error processing line {line_num}: {str(e)}", "ERROR")
                    error_result = {
                        "id": f"line_{line_num}",
                        "prediction": "ERROR",
                        "full_response": f"Error: {str(e)}",
                        "ground_truth": "",
                        "prompt_length": 0,
                        "response_length": 0
                    }
                    results.append(error_result)
        
        # Save results to output file
        output_path = os.environ.get("AZUREML_BI_OUTPUT_PATH", "./")
        output_file_name = Path(file_path).stem
        output_file_path = os.path.join(output_path, output_file_name + "_results.json")
        
        with open(output_file_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print_and_log(f"Saved {len(results)} results to {output_file_path}", "INFO")
    
    return mini_batch

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
def test_batch_requests():
    """
    Test batch requests (batch inference mode) using file-based approach.
    """
    print("\n" + "=" * 60)
    print("TESTING BATCH REQUESTS (File-based Batch Mode)")
    print("=" * 60)
    
    # Sample batch request data
    batch_requests = [
        {
            "id": "batch_test_1",
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
            "max_tokens": 10,
            "temperature": 0.0,
            "ground_truth": "A"
        },
        {
            "id": "batch_test_2", 
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical expert. Read the following USMLE question and choose the best answer. Give me the answer as A/B/C/D/E."
                },
                {
                    "role": "user",
                    "content": "Question:\nWhich of the following is the most effective treatment for bacterial meningitis?\n\nOptions:\nA. Ampicillin\nB. Ceftriaxone\nC. Vancomycin\nD. Azithromycin\nE. Ciprofloxacin"
                }
            ],
            "max_tokens": 10,
            "temperature": 0.0,
            "ground_truth": "B"
        },
        {
            "id": "batch_test_3",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a medical expert. Read the following USMLE question and choose the best answer. Give me the answer as A/B/C/D/E."
                },
                {
                    "role": "user",
                    "content": "Question:\nWhat is the first-line treatment for hypertension in most patients?\n\nOptions:\nA. Beta-blockers\nB. ACE inhibitors\nC. Calcium channel blockers\nD. Diuretics\nE. ARBs"
                }
            ],
            "max_tokens": 10,
            "temperature": 0.0,
            "ground_truth": "D"
        }
    ]
    
    try:
        # Create a temporary JSONL file for testing
        test_file_path = "test_batch_input.jsonl"
        
        # Write test data to JSONL file
        with open(test_file_path, 'w') as f:
            for request in batch_requests:
                f.write(json.dumps(request) + '\n')
        
        print(f"Created test file: {test_file_path}")
        print(f"Test file contains {len(batch_requests)} requests")
        
        # Set environment variable for output (use current directory for testing)
        os.environ["AZUREML_BI_OUTPUT_PATH"] = "./"
        
        # Test batch processing with file paths
        mini_batch = [test_file_path]
        processed_files = run(mini_batch)
        
        print(f"\nBatch Processing Results:")
        print(f"Processed files: {processed_files}")
        
        # Read and display the results
        output_file_path = "test_batch_input_results.json"
        if os.path.exists(output_file_path):
            with open(output_file_path, 'r') as f:
                results = json.load(f)
            
            print(f"\nResults from {output_file_path}:")
            print(f"Number of results: {len(results)}")
            
            correct = 0
            total = len(results)
            
            for i, result in enumerate(results):
                print(f"\n--- Result {i+1} ---")
                print(f"ID: {result.get('id', 'N/A')}")
                print(f"Prediction: {result.get('prediction', 'N/A')}")
                print(f"Ground Truth: {result.get('ground_truth', 'N/A')}")
                print(f"Full Response: {result.get('full_response', 'N/A')[:100]}...")
                
                # Check accuracy
                pred = result.get('prediction', '').strip().upper()
                truth = result.get('ground_truth', '').strip().upper()
                if pred == truth:
                    correct += 1
                    print(f"Correct!")
                else:
                    print(f"Incorrect (Expected: {truth}, Got: {pred})")
            
            accuracy = correct / total if total > 0 else 0
            print(f"\nBatch Accuracy: {correct}/{total} = {accuracy:.2%}")
            
            # Cleanup test files
            try:
                os.remove(test_file_path)
                os.remove(output_file_path)
                print(f"\nCleaned up test files")
            except:
                print(f"\nCould not clean up test files")
        else:
            print(f"Results file not found: {output_file_path}")

    except Exception as e:
        print(f"Error in batch request test: {e}")
        import traceback
        traceback.print_exc()        
        # Cleanup on error
        try:
            if 'test_file_path' in locals():
                os.remove(test_file_path)
            if os.path.exists("test_batch_input_results.json"):
                os.remove("test_batch_input_results.json")
        except:
            pass


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
    # Run batch request test
    test_batch_requests()