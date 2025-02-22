from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM

# Initialize the model and tokenizer
print("check1")
model_name = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("check2")
model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
print("check3")
# Prepare input text
text = "Write a short story about a robot learning to paint:"
inputs = tokenizer(text, return_tensors="pt", max_length=1024, padding=True)["input_ids"]
print("check4")
inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"]
print("check5")
outputs = model.generate(inputs, max_new_tokens=5)
print("check6")
decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded_text)
