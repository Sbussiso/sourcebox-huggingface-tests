import os
import sys
import time
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# Load environment variables from .env file
load_dotenv()

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define the model to use
# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "gpt2"
# model_name = "distilgpt2"
# model_name = "mistralai/Mistral-7B-v0.1"
# model_name = "microsoft/Phi-3.5-MoE-instruct"

model_safe_name = model_name.replace("/", "_")

# Path to save the model after first download
model_save_path = f"./_{model_safe_name}_"

# Download the model and save it with safe serialization
if not os.path.exists(model_save_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=api_token,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.save_pretrained(model_save_path, safe_serialization=True)
else:
    # Load the locally saved model in eval mode for faster inference
    model = AutoModelForCausalLM.from_pretrained(
        model_save_path,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_token)

# Create the pipeline for text generation using PyTorch
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",  # Use available device (CPU)
)

def model_response(prompt: str) -> str:
    """
    Generate a response from the model.
    """
    try:
        # Start the timer to monitor processing time
        start_time = time.time()

        # Generate a response with optimized parameters
        sequences = text_gen_pipeline(
            prompt,
            do_sample=True,  # Keep sampling enabled for diverse outputs
            max_length=128,  # Increase max_length for more complete responses
            truncation=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # End the timer
        end_time = time.time()
        
        # Extract the generated text
        generated_text = sequences[0]['generated_text'].strip()

        # Return the generated response
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        return f"Chatbot: {generated_text}"

    except Exception as e:
        return f"An error occurred: {e}"


print("\n____________Test1_____________")
# Example prompt to test the model
prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?'
print(model_response(prompt))
# sys.exit()

print("\n____________Test2_____________")
# Prompt to test model
prompt = 'Explain how photosynthesis works in simple terms.'
print(model_response(prompt))
# sys.exit()

print("\n____________Test3_____________")
# Prompt to test model
prompt = 'What is the capital of Washington state?'
print(model_response(prompt))
# sys.exit()
