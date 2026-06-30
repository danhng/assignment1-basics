from .tokenizer import FastTokenizer
from .fast_bpe_bytes import run_train_bpe
import logging

from ..tests import test_tokenizer

# 1. Create a custom logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('main')
# 2. Create handlers
c_handler = logging.StreamHandler()  # For console
# 3. Create formatters and add to handlers
c_format = logging.Formatter('%(levelname)s - line %(lineno)d - %(message)s')
c_handler.setFormatter(c_format)
c_handler.setLevel(logging.DEBUG)
# 4. Add handlers to the logger
logger.addHandler(c_handler)

def testTokenizer(): 
    splitTextToken = "<|endoftext|>"
    specialTokens = []
    dataset = "test.txt"
    dataset = "TinyStoriesV2-GPT4-valid.txt"
    # vocab, merges = run_train_bpe(f"assignment1-basics/data/{dataset}", 
    #             output_path=f"assignment1-basics/data/output/{dataset}", 
    #             vocab_size=1000, special_tokens=specialTokens, split_text_token=splitTextToken, 
    #             chunk_size_to_process=100*1024*1024, 
    #             get_max_by_cache=True, get_init_multi_process=True, process_count = 4, outputMergeJson=False)
    
    vocabPath = "assignment1-basics/data/output/TinyStoriesV2-GPT4-valid.txt-bytes-cTrue-4-1000-260630221335-7.2-vocab.json"
    mergesPath = "assignment1-basics/data/output/TinyStoriesV2-GPT4-valid.txt-bytes-cTrue-4-1000-260630221335-7.2-merges.json"
    tokenizerr = FastTokenizer.from_files(vocab_filepath=vocabPath, merges_filepath=mergesPath, special_tokens=[splitTextToken], inputFormatJson=False)
    # diff = DeepDiff(vocab, tokenizerr.vocab_id_word)
    # print(diff)
    # assert vocab == tokenizerr.vocab_id_word
    # assert merges == tokenizerr.merges

    all_ids = []
    with open("assignment1-basics/data/tinystories_sample.txt") as f:
        for _id in tokenizerr.encode_iterable(f):
            all_ids.append(_id)
    
    test = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    # test = "Héllò hôw <|endoftext|><|endoftext|> are ü?<|endoftext|>"
    # test = "🙃"
    encoded = tokenizerr.encode(test)
    decoded = tokenizerr.decode(encoded)
    logger.debug(f"{test} encoded -> {encoded}")
    logger.debug(f"{encoded} decoded -> {tokenizerr.decode(encoded)}")
    assert test == decoded


    ## Usage
if __name__ == '__main__':
   test_tokenizer.test_address_matches_tiktoken()
#    testTokenizer()