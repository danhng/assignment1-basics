import os
import math
import time
from typing import BinaryIO
import regex as re

import io
import logging

from tqdm import tqdm

# 1. Create a custom logger
logger = logging.getLogger('fast_bpe')
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

def find_chunk_boundaries(
    file: BinaryIO,
    chunk_size: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # chunk_size = file_size // desired_num_chunks
    desired_num_chunks = math.ceil(file_size / chunk_size)

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in tqdm(range(len(chunk_boundaries)), desc="find_chunk_boundaries"):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    chunk_boundaries.append(0)
    return sorted(set(chunk_boundaries))

"""
    Returns: dict(tuple[str] -> count): The pretoken -> count
"""
def initPretoken(inputPath: str | os.PathLike, splitTokens: str, specialTokens: list[str], numProcesses = 1, chunkSizeToProcess = 1024*1024): 
    map = {}
    """
    iterate through chunks (splitted by tokens)
    """
    with open(inputPath, "rb") as f:
        delimTokens = specialTokens + [splitTokens]
        boundaries = find_chunk_boundaries(f, chunkSizeToProcess, bytes(splitTokens, "utf-8"))
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in tqdm(zip(boundaries[:-1], boundaries[1:]), desc="Init Pretoken"):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore").strip()
            splitTokenRegex = r"|"+"|".join(re.escape(escapedToken) for escapedToken in(delimTokens))
            chunk = re.sub(splitTokenRegex, "", chunk)
            if chunk:
                # remove special tokens
                regexPretoken = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                matches = re.finditer(regexPretoken, chunk)
                for match in matches: 
                    word = match.group()
                    chars = tuple(word) 
                    map[chars] = map.get(chars, 0) + 1
        return map

"""
Give the pretokens, count the adjacent pairs count 
return map tuple[str, str] -> count
{('u',): 1, (' ', 'd', 'o', 'n'): 1, ("'", 't'): 1 ...
pair cache (map of map): {(o, w) -> {(l, (o, w)) -> 3}, ...}}
"""
def getMaxPairCount(pretokens, initPairCache = False, pairCache = {}): 
    pairs = {}
    maxPairCount = 0
    maxPair = tuple()
    for word, count in pretokens.items(): 
        # zip to create pairs. 
        for i in range(len(word) - 1): 
            targetPair = tuple([''.join(word[i]), ''.join(word[i+1])]) # todo flatten tuple bytes
            # todo: iterate through all pair cache instead of words.
            pairs[targetPair] = pairs.get(targetPair, 0) + count
            # compare max and lexicographic
            if (pairs[targetPair] > maxPairCount) or ((pairs[targetPair] == maxPairCount) and (targetPair > maxPair)):
                maxPairCount = pairs[targetPair]
                maxPair = targetPair
            if initPairCache: 
                if targetPair not in pairCache: 
                    pairCache[targetPair] = {}
                pairCache[targetPair][word] = count
                    
    return tuple([maxPair, maxPairCount]), pairCache

"""
    return new pre token after maxpair is merged
"""
def mergePretoken(pretokens, maxPair):
    # loop through all pretoken
    # replace pair of bytes with maxPair
    newPreTokens = dict()
    for (pretoken, count) in pretokens.items(): 
        newPretoken = []
        i = 0
        while i < (len(pretoken)): 
            if i < len(pretoken)-1 and tuple([''.join(pretoken[i]), ''.join(pretoken[i+1])]) == maxPair[0]: 
                newPretoken.append(maxPair[0])
                # todo: reindex pair cache after merge
                i=i+2 # advance past current pair
            else:
                newPretoken.append(pretoken[i])
                i = i+1
        # logger.debug(f"new merged token: {pretoken} -> {newPretoken}")
        newPreTokens[tuple(newPretoken)] = count
    return newPreTokens


"""
    return new pre token after maxpair is merged
"""
def mergePretokenCache(pretokens, maxPair, pairCache):
    # loop through all pretoken
    # replace pair of bytes with maxPair
    maxPairBytes = maxPair[0]
    wordsToProcess = pairCache[maxPairBytes]
    for word in wordsToProcess: 
        # create the cache for max pair (maxword -> {word: count})
        newPretoken = []
        i = 0
        while i < len(word): 
            if i < len(word)-1 and tuple([''.join(word[i]), ''.join(word[i+1])]) == maxPair[0]: 
                newPretoken.append(maxPair[0])
                # todo: reindex pair cache after merge
                i=i+2 # advance past current pair
            else:
                newPretoken.append(word[i])
                i = i+1
        
        newPretoken = tuple(newPretoken)
        # iterate through pair in new Pretoken to do the recalculation
        for i in range(len(word) - 1): 
            newPretokenPair = tuple([''.join(word[i]), ''.join(word[i+1])])
            if newPretokenPair == maxPair[0]: 
                if i > 0:
                    newPair = tuple([''.join(word[i-1]), ''.join(maxPair[0])]) # h, ow
                    oldAdjacentPair = tuple([''.join(word[i-1]), ''.join(word[i])]) # h, o
                    if newPair not in pairCache: 
                        pairCache[newPair] = {}
                    pairCache[newPair][newPretoken] = pairCache[oldAdjacentPair][word] # h, ow -> {(h, ow) -> 1} 
                    del pairCache[oldAdjacentPair][word] # remove the old pair cache. Ex. {h, o -> {(h, o, w) -> 1, (h, o, t) -> 1}} => {h, o -> {(h, o, t) -> 1}}
                    if not pairCache[oldAdjacentPair]: 
                        del pairCache[oldAdjacentPair]
                if i < len(word) - 2:
                    newPair = tuple([''.join(maxPair[0]), ''.join(word[i+2])]) # ow, e
                    oldAdjacentPair = tuple([''.join(word[i+1]), ''.join(word[i+2])]) # w, e
                    if newPair not in pairCache: 
                        pairCache[newPair] = {}
                    pairCache[newPair][newPretoken] = pairCache[oldAdjacentPair][word] # ow, e -> {(l, ow, e, r) -> 1} 
                    del pairCache[oldAdjacentPair][word] # remove the old pair cache. Ex. {w, e -> {(l, o, w, e, r) -> 1, (w, e, s, t) -> 1}} => {w, e -> {(w, e, s, t) -> 1}}
                    if not pairCache[oldAdjacentPair]: 
                        del pairCache[oldAdjacentPair]
            elif (i < len(word) - 2 and maxPair[0] != tuple([''.join(word[i+1]), ''.join(word[i+2])])) and (i > 1 and maxPair[0] != tuple([''.join(word[i-2]), ''.join(word[i-1])])): 
                pairCache[newPretokenPair][newPretoken] = pairCache[newPretokenPair][word] 
                del pairCache[newPretokenPair][word] 
        pretokens[newPretoken] = pretokens[word]         # add new preToken to pretokenMap
        del pretokens[word] # remove word h, o, w
    del pairCache[maxPair[0]] # remove the old pair (as they are merged to one token), o, w -> {}
    # remove maxpair as pair from the cache after merge
    return pretokens

"""Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
"""
    
"""
    1. initialize vocab
    2. initialize pretoken from text
    3. for each round
        get pair with greatest counts in pretoken
        add most frequent pair to vocab
        merge new token into pretoken
"""
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    split_text_token = "<|endoftext|>",
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    start_time = time.perf_counter()
    pretokens = initPretoken(inputPath=input_path, splitTokens=split_text_token,specialTokens=special_tokens)
    end_time1 = time.perf_counter()
    elapsed_time1 = end_time1 - start_time
    logger.info(f"Init pretokens: Vocab size: {vocab_size}, Elapsed time: {elapsed_time1:.4f} seconds")
    logger.info(f"Init pretokens: {pretokens }")
    # logger.debug(f"init pretoken: {pretokens}")
    vocab = {}
    merges = []
    pairCache = {}
    for iteration in tqdm(range(vocab_size), desc="Training vocab"): 
        logger.debug(f"iteration {iteration}")
        maxPair, pairCache = getMaxPairCount(pretokens, iteration == 0, pairCache)
        pretokens = mergePretokenCache(pretokens, maxPair, pairCache)
        logger.debug(pairCache)
        logger.debug(pretokens)
        logger.info(f"iter {iteration}: max pair: {maxPair}")
        logger.info(f"iter {iteration}: pretokens: {pretokens}")
        if (maxPair[1] > 0):
            vocab[len(vocab)] = maxPair[0]
            merges.append(maxPair[0])
    #add special token to vocab
    for specialToken in special_tokens: 
        vocab[len(vocab)] = specialToken
    end_time2 = time.perf_counter()
    elapsed_time2 = end_time2 - start_time
    logger.info(f"Vocab size: {vocab_size}, Elapsed time: {elapsed_time2:.4f} seconds")
    with open(f"{input_path}-{vocab_size}.txt", "w") as file:
        for merge in merges: 
            file.write(f"{merge[0]}-{merge[1]}\n")
    return vocab, merges

## Usage
splitTextToken = "<|endoftext|>"
specialTokens = []
run_train_bpe("assignment1-basics/data/test.txt", 6, specialTokens, splitTextToken)
# print(initPretoken("data/test.txt", splitTextToken, specialTokens))