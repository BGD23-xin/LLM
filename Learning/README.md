## Definition

LLM: large language model

RAG: Retrieval Augmented Generation

## Use in the saturn cloud 

```bash
https://huggingface.co/google/flan-t5-xl
```

The follow codes came from the previous link,which are suitable for GPU
```python
# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))

```
