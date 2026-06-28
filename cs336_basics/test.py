import fast_bpe_bytes as fastBpeBytes
import fast_bpe_string as fastBpeString
import tokenizer as tokenizer
import logging

# 1. Create a custom logger
logger = logging.getLogger('test')
logger.setLevel(logging.DEBUG)

# 2. Create handlers
c_handler = logging.StreamHandler()  # For console
f_handler = logging.FileHandler('app.log', mode='w')  # For file

# 3. Create formatters and add to handlers
c_format = logging.Formatter('%(levelname)s - line %(lineno)d - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(lineno)d - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# 4. Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

from deepdiff import DeepDiff

## Usage
if __name__ == '__main__':
    splitTextToken = "<|endoftext|>"
    specialTokens = []
    # dataset = "test.txt"
    dataset = "TinyStoriesV2-GPT4-valid.txt"
    # vocab, merges = fastBpeBytes.run_train_bpe(f"assignment1-basics/data/{dataset}", 
    #             output_path=f"assignment1-basics/data/output/{dataset}", 
    #             vocab_size=1000, special_tokens=specialTokens, split_text_token=splitTextToken, 
    #             chunk_size_to_process=100*1024*1024, 
    #             get_max_by_cache=True, get_init_multi_process=True, process_count = 8)
    
    vocabPath = f"assignment1-basics/data/output/TinyStoriesV2-GPT4-valid.txt-bytes-cTrue-8-1000-260628143154-7.7-vocab.json"
    mergesPath = f"assignment1-basics/data/output/TinyStoriesV2-GPT4-valid.txt-bytes-cTrue-8-1000-260628143154-7.7-merges.json"
    tokenizerr = tokenizer.FastTokenizer.from_files(vocab_filepath=vocabPath, merges_filepath=mergesPath, special_tokens=[splitTextToken])
    # diff = DeepDiff(vocab, tokenizerr.vocab_id_word)
    # print(diff)
    # assert vocab == tokenizerr.vocab_id_word
    # assert merges == tokenizerr.merges
    wordTest = "passive<|endoftext|>active"
    logger.info(f"{wordTest} -> {tokenizerr.encode(wordTest)}")
    logger.info(f"{wordTest} -> {[ tokenizerr.vocab_id_word[id] for id in tokenizerr.encode(wordTest)]}")

    # <\|endoftext\|>|'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
