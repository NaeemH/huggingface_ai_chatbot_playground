import os
from huggingface_hub import hf_hub_download

# https://github.com/mneedham/LearnDataWithMark/blob/main/llm-own-laptop/notebooks/LLMOwnLaptop.ipynb

HUGGING_FACE_API_KEY = input('ENTER HUGGING FACE API KEY: ')

model_id = "lmsys/fastchat-t5-3b-v1.0"
filenames = [
        "pytorch_model.bin", "added_tokens.json", "config.json", "generation_config.json", 
        "special_tokens_map.json", "spiece.model", "tokenizer_config.json"
]
for filename in filenames:
        downloaded_model_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    token=HUGGING_FACE_API_KEY
        )
        print(downloaded_model_path)

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipeline = pipeline("text2text-generation", model=model, device=-1, tokenizer=tokenizer, max_length=1000)