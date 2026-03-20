"""Sample usage of the trained model."""

from Titan_simple.src import model
from Titan_simple.src.inference import TitansFastGenerator
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "The future of artificial intelligence is"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generator = TitansFastGenerator(model)

output = generator.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40
)

print(tokenizer.decode(output[0]))