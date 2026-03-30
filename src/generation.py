"""Generation script using trained TitansMAC model on random wikitext prompts."""

import torch
import random
from datasets import load_dataset
from transformers import AutoTokenizer

from model import TitansMAC
from config import TitansConfig


def generate_completion(model, prompt_text, tokenizer, device, max_new_tokens=50, temperature=0.8, top_k=40):
    """Generate text completion for a given prompt."""
    model.eval()
    
    with torch.no_grad():
        # Tokenize prompt
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        current_tokens = input_ids.clone()
        states = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, states = model(current_tokens, states)
            
            # Get last token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            
            current_tokens = torch.cat([current_tokens, next_token], dim=-1)
            
            # Detach states
            if states is not None:
                states = [s.detach() for s in states]
    
    return tokenizer.decode(current_tokens[0])


def get_random_prompts(tokenizer, num_prompts=5, prompt_length=15):
    """Extract random prompts from wikitext dataset."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Filter for non-empty texts
    texts = [ex["text"] for ex in dataset["validation"] if len(ex["text"]) > 50]
    
    prompts = []
    for _ in range(num_prompts):
        # Sample random text
        text = random.choice(texts)
        # Tokenize and get substring
        tokens = tokenizer.tokenize(text)
        if len(tokens) >= prompt_length:
            prompt_tokens = tokens[:prompt_length]
            prompt_text = tokenizer.convert_tokens_to_string(prompt_tokens)
            prompts.append(prompt_text)
        else:
            prompts.append(text)
    
    return prompts


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load config and model
    config = TitansConfig(
        vocab_size=50257,
        dim=768,
        num_layers=8,
        chunk_size=512,
        dropout=0.1
    )
    
    model = TitansMAC(config).to(device)
    checkpoint_path = "titans_mac_padded.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Model loaded from {checkpoint_path}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Get random prompts
    print("Sampling random prompts from wikitext...")
    prompts = get_random_prompts(tokenizer, num_prompts=5, prompt_length=15)
    
    print("\n" + "="*80)
    print("GENERATION RESULTS")
    print("="*80 + "\n")
    
    # Generate completions
    for i, prompt in enumerate(prompts, 1):
        print(f"[Example {i}]")
        print(f"Prompt: {prompt}")
        
        completion = generate_completion(
            model, 
            prompt, 
            tokenizer, 
            device,
            max_new_tokens=50,
            temperature=0.8,
            top_k=40
        )
        
        print(f"Completion: {completion}")
        print("-" * 80 + "\n")


if __name__ == "__main__":
    main()