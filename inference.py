from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from transformers import pipeline

model = GPT2LMHeadModel.from_pretrained('output')

generator = pipeline(model=model)


def data():
    for i in range(4):
        yield f"My example {i}"


pipe = pipeline(model="gpt2", device=0)
generated_characters = 0
for out in pipe(data()):
    generated_characters += len(out[0]["generated_text"])

print(generated_characters)