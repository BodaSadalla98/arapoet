import torch
device = 0 if torch.cuda.is_available() else 'cpu'

print('Using device:', device)

from transformers import GPT2TokenizerFast, pipeline
#for base and medium
from transformers import GPT2LMHeadModel
#for large and mega
# pip install arabert
from arabert.aragpt2.grover.modeling_gpt2 import GPT2LMHeadModel

from arabert.preprocess import ArabertPreprocessor

MODEL_NAME='/home/abdelrahman.atef/boda/ara-poet/output/checkpoint-570580'
arabert_prep = ArabertPreprocessor(model_name=MODEL_NAME)


model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
generation_pipeline = pipeline("text-generation",model=model,tokenizer=tokenizer, device=device)

text="كان يا ما كان في قديم"
text_clean = arabert_prep.preprocess(text)
print(text_clean)
print(generation_pipeline(text_clean))

