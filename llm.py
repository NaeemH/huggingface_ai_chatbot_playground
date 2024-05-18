import os
from huggingface_hub import hf_hub_download

#HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")
HUGGING_FACE_API_KEY = input('ENTER HUGGING FACE API KEY: ')
"""
model_id = "lmsys/fastchat-t5-3b-v1.0"
filenames = [
        "pytorch_model.bin", "added_tokens.json", "config.json", "generation_config.json", 
        "special_tokens_map.json", "spiece.model", "tokenizer_config.json"
]
"""
filename = input('ENTER FILENAME WITH EXT FROM REPO: ')
file_list = []
model_name = ''
with open(filename) as file:
    # read first line of file as model_name
    model_name = file.readline()
    # create list of elements based on filename
    file_list = [line.rstrip() for line in file]
print('LLM MODEL: ', model_name)
print('LIST OF FILES: ', file_list)

for filename in file_list:
        downloaded_model_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    token=HUGGING_FACE_API_KEY
        )
        print(downloaded_model_path)