import numpy as np
import logging
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

def testTokenizer(): 
    splitTextToken = "<|endoftext|>"
    specialTokens = []
    dataset = "test.txt"
    dataset = "TinyStoriesV2-GPT4-train.txt"
    # vocab, merges = run_train_bpe(f"assignment1-basics/data/{dataset}", 
    #             output_path=f"assignment1-basics/data/output/{dataset}", 
    #             vocab_size=10000, special_tokens=specialTokens, split_text_token=splitTextToken, 
    #             chunk_size_to_process=100*1024*1024, 
    #             get_max_by_cache=True, get_init_multi_process=True, process_count = 4, outputMergeJson=False)
  
    datasetEncode = "tinystories_sample_5M.txt"
    vocabPath = "data/output/TinyStoriesV2-GPT4-train.txt-bytes-cTrue-4-10000-260703145356-219.7-vocab.json"
    mergesPath = "data/output/TinyStoriesV2-GPT4-train.txt-bytes-cTrue-4-10000-260703145356-219.7-merges.json"
    tokenizerr = FastTokenizer.from_files(vocab_filepath=vocabPath, merges_filepath=mergesPath, special_tokens=[splitTextToken], vocab_file_json=False)
    tokenizerr.serialize_encode(f"data/{datasetEncode}", f"data/output/{datasetEncode}-encoded.npy")

## Usage
if __name__ == '__main__':
    testTokenizer()