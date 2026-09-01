import sys
import tomllib
import numpy as np
import logging
from cs336_basics.fast_bpe_bytes import run_train_bpe
from cs336_basics.tokenizer import FastTokenizer
import linear

# 1. Create a custom logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
# # 2. Create handlers
# c_handler = logging.StreamHandler()  # For console
# # 3. Create formatters and add to handlers
# c_format = logging.Formatter('%(levelname)s - line %(lineno)d - %(message)s')
# c_handler.setFormatter(c_format)
# c_handler.setLevel(logging.DEBUG)
# # 4. Add handlers to the logger
# logger.addHandler(c_handler)

def train_bpe(config): 
    dataset = "TinyStoriesV2-GPT4-train.txt"
    splitTextToken = config.get("splitTextToken")
    specialTokens = config.get("specialTokens")
    vocab_size = config.get("vocab_size")
    run_train_bpe(f"data/{dataset}", 
                output_path=f"data/output/{dataset}", 
                vocab_size=vocab_size, special_tokens=specialTokens, split_text_token=splitTextToken, 
                chunk_size_to_process=int(config.get("chunk_size_to_process_mb")*1024*1024), 
                get_max_by_cache=True, get_init_multi_process=True, process_count = config.get("processor_count"), output_merges_json=False)
    

def tokenize_text(config): 
    # "<|endoftext|>"
    splitTextToken = config.get("splitTextToken")
    specialTokens = config.get("specialTokens")
    # "tinystories_sample_5M.txt"
    datasetEncode = config.get("datasetEncode")
    vocabPath = config.get("vocabPath")
    mergesPath = config.get("mergesPath")
    tokenizerr = FastTokenizer.from_files(vocab_filepath=vocabPath, merges_filepath=mergesPath, special_tokens=[splitTextToken], vocab_file_json=False)
    tokenizerr.serialize_encode(f"data/{datasetEncode}", f"data/output/{datasetEncode}-encoded-{sys.platform}.npy")

## Usage
if __name__ == '__main__':
    with open("config/tokenizer.toml", "rb") as file:
        # bpe_training_config = tomllib.load(file)
        tokenizer_config = tomllib.load(file)
    tokenize_text(config=tokenizer_config)
    # train_bpe(bpe_training_config)