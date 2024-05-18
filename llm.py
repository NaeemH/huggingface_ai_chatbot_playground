import os
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

#HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")
HUGGING_FACE_API_KEY = input('ENTER HUGGING FACE API KEY: ')
model_id = "lmsys/fastchat-t5-3b-v1.0"
filenames = [
        "pytorch_model.bin", "added_tokens.json", "config.json", "generation_config.json", 
        "special_tokens_map.json", "spiece.model", "tokenizer_config.json"
]
#filename = input('ENTER FILENAME WITH EXT FROM REPO: ')
#file_list = []
print('LLM MODEL: ', model_id)
print('LIST OF FILES: ', filenames)

for filename in filenames:
        downloaded_model_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    token=HUGGING_FACE_API_KEY
        )
        print(downloaded_model_path)


# Tokenize and use data
tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipeline = pipeline("text2text-generation", model=model, device=-1, tokenizer=tokenizer, max_length=1000)