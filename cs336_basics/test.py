import fast_bpe_bytes as fastBpeBytes
import fast_bpe_string as fastBpeString
import tokenizer as tokenizer
import logging

# 1. Create a custom logger
logger = logging.getLogger('fast_bpe_bytes')
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

## Usage
if __name__ == '__main__':
    splitTextToken = "<|endoftext|>"
    specialTokens = []
    dataset = "test.txt"
    # dataset = "corpus.en"
    vocab, merges = fastBpeBytes.run_train_bpe(f"assignment1-basics/data/{dataset}", 
                output_path=f"assignment1-basics/data/output/{dataset}", 
                vocab_size=500, special_tokens=specialTokens, split_text_token=splitTextToken, 
                chunk_size_to_process=100*1024*1024, 
                get_max_by_cache=True, get_init_multi_process=True, process_count = 4)

    tokenizerr = tokenizer.FastTokenizer(vocab, merges, None)
    logger.debug(tokenizerr.encode("owne"))