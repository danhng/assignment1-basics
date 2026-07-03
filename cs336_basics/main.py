from deepdiff import DeepDiff

from .tokenizer import FastTokenizer
from .fast_bpe_bytes import run_train_bpe
import logging

from ..tests import test_tokenizer

# 1. Create a custom logger
logging.basicConfig(level=logging.DEBUG)
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
    dataset = "TinyStoriesV2-GPT4-valid.txt"
    vocab, merges = run_train_bpe(f"assignment1-basics/data/{dataset}", 
                output_path=f"assignment1-basics/data/output/{dataset}", 
                vocab_size=1000, special_tokens=specialTokens, split_text_token=splitTextToken, 
                chunk_size_to_process=100*1024*1024, 
                get_max_by_cache=True, get_init_multi_process=True, process_count = 4, outputMergeJson=False)
    
  
    
    # vocabPath = "assignment1-basics/data/gpt2_vocab.json"
    # mergesPath = "assignment1-basics/data/gpt2_merges.txt"
    # tokenizerr = test_tokenizer.get_tokenizer_from_vocab_merges_path(
    #     vocab_path=vocabPath, merges_path=mergesPath, special_tokens=["<|endoftext|>"]
    # )
    # tokenizerr = FastTokenizer.from_files(vocab_filepath=vocabPath, merges_filepath=mergesPath, special_tokens=[splitTextToken], inputFormatJson=False)
    # diff = DeepDiff(vocab, tokenizerr.vocab_id_word)
    # print(diff)
    # assert vocab == tokenizerr.vocab_id_word
    # assert merges == tokenizerr.merges

    # all_ids = []
    # with open("assignment1-basics/data/tinystories_sample.txt") as f:
    #     for _id in tokenizerr.encode_iterable(f):
    #         all_ids.append(_id)
    
    # test = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    # test = "Hello, how are you?"
    # # test = "Héllò hôw <|endoftext|><|endoftext|> are ü?<|endoftext|>"
    # # test = "🙃"
    # encoded = tokenizerr.encode(test)
    # decoded = tokenizerr.decode(encoded)
    # logger.info(f"{test} encoded -> {encoded}")
    # logger.info(f"{encoded} decoded -> {tokenizerr.decode(encoded)}")
    # [15496, 11, 703, 389, 345, 30]

    # assert test == decoded

def test_tokenizer_full(): 
    test_functions = [
        test_tokenizer.test_roundtrip_empty,
        test_tokenizer.test_address_roundtrip,
        test_tokenizer.test_address_matches_tiktoken,
        test_tokenizer.test_ascii_string_matches_tiktoken,
        test_tokenizer.test_empty_matches_tiktoken,
        test_tokenizer.test_encode_iterable_memory_usage,
        test_tokenizer.test_encode_iterable_tinystories_matches_tiktoken,
        test_tokenizer.test_encode_iterable_tinystories_sample_roundtrip,
        test_tokenizer.test_encode_memory_usage,
        test_tokenizer.test_encode_special_token_double_newline_non_whitespace,
        test_tokenizer.test_encode_special_token_trailing_newlines,
        test_tokenizer.test_german_matches_tiktoken,
        test_tokenizer.test_german_roundtrip,
        test_tokenizer.test_roundtrip_unicode_string,
        test_tokenizer.test_roundtrip_single_character,
        test_tokenizer.test_roundtrip_unicode_string_with_special_tokens,
        test_tokenizer.test_roundtrip_ascii_string,
        test_tokenizer.test_overlapping_special_tokens,
        test_tokenizer.test_tinystories_matches_tiktoken,
        test_tokenizer.test_tinystories_sample_roundtrip,
        test_tokenizer.test_single_character_matches_tiktoken,
        test_tokenizer.test_single_unicode_character_matches_tiktoken,
        test_tokenizer.test_unicode_string_matches_tiktoken,
        test_tokenizer.test_unicode_string_with_special_tokens_matches_tiktoken,
    ]
    passed = 0
    failed = []
    for test_function in test_functions: 
        try: 
            test_function()
            passed = passed + 1
        except: 
            failed.append(test_function)
            logger.exception("A critical error occurred while executing ", test_function)
    logger.info(f"Pass {passed}/{len(test_functions)}")
    logger.info(f"Failed: {[failFunction.__name__ for failFunction in failed]}")
    return

## Usage
if __name__ == '__main__':
    # test_tokenizer_full()
    test_tokenizer.test_encode_memory_usage()
    # test_tokenizer.test_encode_special_token_double_newline_non_whitespace()
    # test_tokenizer.test_roundtrip_unicode_string_with_special_tokens()
    # test_tokenizer.test_unicode_string_with_special_tokens_matches_tiktoken()
    # test_tokenizer.test_roundtrip_empty()
    # test_tokenizer.test_roundtrip_empty()
    # test_tokenizer.test_roundtrip_empty()
#    testTokenizer()
