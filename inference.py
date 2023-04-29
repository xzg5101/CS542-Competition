from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from transformers import pipeline

config_class, model_class, tokenizer_class = GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

model = model_class.from_pretrained("output")

print(model._get_name())

generator = pipeline(model=model)


def data():
    for i in range(4):
        yield f"My example {i}"


pipe = pipeline(model="gpt2", device=0)
generated_characters = 0
for out in pipe(data()):
    generated_characters += len(out[0]["generated_text"])

print(generated_characters)