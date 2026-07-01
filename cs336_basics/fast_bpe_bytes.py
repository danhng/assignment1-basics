from datetime import datetime
import multiprocessing
from operator import itemgetter
import os
import math
import time
from typing import BinaryIO
import regex as re
import logging
from tqdm import tqdm
from sortedcontainers import SortedList
import json

# 1. Create a custom logger
logger = logging.getLogger('fast_bpe_bytes')

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
    logger.info(f"File size: {file_size}")
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
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            splitTokenRegex = r"|"+"|".join(re.escape(escapedToken) for escapedToken in(delimTokens))
            chunk = re.sub(splitTokenRegex, "", chunk)
            if chunk:
                # remove special tokens
                regexPretoken = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                matches = re.finditer(regexPretoken, chunk)
                for match in matches: 
                    word = match.group()
                    wordBytes = tuple(bytes([b]) for b in word.encode("utf-8")) # string to tuple of bytes
                    map[wordBytes] = map.get(wordBytes, 0) + 1
        return map

def initEncodedBasedVocab(): 
    dictEncoded = {}
    # iterate through first 256 possible values
    for b in range(0,256): 
        if 33 <= int(b) <= 126 or 161 <= int(b) <= 172 or 174 <= int(b): 
            dictEncoded[b] = chr(b)
    i = 0
    for b in range(0,256): 
        if not (33 <= int(b) <= 126 or 161 <= int(b) <= 172 or 174 <= int(b)): 
            dictEncoded[b] = chr(256+i)
            i = i+1
    return dictEncoded

def initEncodedBasedVocabReverse(baseVocab): 
    reverse = {value:key for key, value in baseVocab.items()}
    return reverse

BASE_VOCAB_BYTE_WORD = initEncodedBasedVocab()
BASE_VOCAB_WORD_BYTE = initEncodedBasedVocabReverse(BASE_VOCAB_BYTE_WORD)

def bytesToShiftedUnicode(wordBytes, merge=False): 
    outputChars = [BASE_VOCAB_BYTE_WORD[b] for b in wordBytes]
    if merge:
        return ''.join(outputChars)
    return tuple(outputChars)

"""
    Returns: dict(tuple[str] -> count): The pretoken -> count
"""
def processChunk(task): 
    start, end, inputPath, splitTokens, specialTokens = task
    delimTokens = specialTokens + [splitTokens]
    map = {}
    with open(inputPath, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")        
        splitTokenRegex = r"|"+"|".join(re.escape(escapedToken) for escapedToken in(delimTokens))
        chunk = re.sub(splitTokenRegex, "", chunk)
        if chunk:
            # remove special tokens
            regexPretoken = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            matches = re.finditer(regexPretoken, chunk)
            for match in matches: 
                word = match.group().encode("utf-8") # todo convert from bytes to 256 shifted unicode characters
                # word = word.replace(chr(0x20), chr(288)).replace(chr(0x0A), chr(266))
                # wordBytes = tuple(bytes([b]) for b in word) # string to tuple of bytes
                wordEncoded = bytesToShiftedUnicode(word) # tuple of chars
                map[wordEncoded] = map.get(wordEncoded, 0) + 1
    return map


def initPretokenMultiProcess(inputPath: str | os.PathLike, splitTokens: str, specialTokens: list[str], chunkSizeToProcess = 1024*1024, processCount = 1): 
    # if __name__ == "__main__":
    map = {}
    """
    iterate through chunks (splitted by tokens)
    """
    with open(inputPath, "rb") as f:
        boundaries = find_chunk_boundaries(f, chunkSizeToProcess, bytes(splitTokens, "utf-8"))
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
            
        #create task
        tasks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((start, end, inputPath, splitTokens, specialTokens))
            
        with multiprocessing.Pool(processes=processCount) as pool:
        # pool.map distributes the list items across worker processes
            iterMap = tqdm(pool.imap_unordered(processChunk, tasks), total=len(boundaries)-1, desc="Init Pretoken")
            # submaps = pool.map(processChunk, tasks)
            print(f"Total workers in pool: {processCount}")  # Outputs: 4
            for submap in iterMap: 
                for word, count in submap.items(): 
                    map[word] = map.get(word, 0) + count
        logger.debug(f"Init pre token size: {len(map)}")
        logger.debug(f"pre token: {map}")
    return map


"""
Give the pretokens, count the adjacent pairs count 
return map tuple[str, str] -> count
{('u',): 1, (' ', 'd', 'o', 'n'): 1, ("'", 't'): 1 ...
pair cache (map of map): {(o, w) -> {(l, (o, w)) -> [<number of words>, <number of occurences>]}, ...}}
"""
def getMaxPairCountCache2(pretokens, initPairCache = False, pairCache = {}, pairCacheSort = SortedList()): 
    pairs = {}
    maxPairCount = 0
    maxPair = tuple()
    
    # if not init pair cache, do the slow way
    if initPairCache: 
        for wordEncodedChars, wordCount in pretokens.items(): 
            # zip to create pairs. 
            for i in range(len(wordEncodedChars) - 1): 
                targetPair = tuple([wordEncodedChars[i], wordEncodedChars[i+1]]) # no need join because this is the first round
                # todo: iterate through all pair cache instead of words.
                pairs[targetPair] = pairs.get(targetPair, 0) + wordCount
                # compare max and lexicographic
                if (pairs[targetPair] > maxPairCount) or ((pairs[targetPair] == maxPairCount) and (targetPair > maxPair)):
                    maxPairCount = pairs[targetPair]
                    maxPair = targetPair
                # init pair cache
                if targetPair not in pairCache: 
                    pairCache[targetPair] = {}
                if wordEncodedChars not in pairCache[targetPair]: 
                    pairCache[targetPair][wordEncodedChars] = [0, 0]
                pairCache[targetPair][wordEncodedChars][0] = wordCount # append count to handle to multi-pair in a word
                pairCache[targetPair][wordEncodedChars][1] = pairCache[targetPair][wordEncodedChars][1] + 1 # append count to handle to multi-pair in a word
        # init pairCacheSort
        for targetPair, pairSpec in pairCache.items(): 
            pairCount = sum(pairSpecMap[0] * pairSpecMap[1] for pairSpecMap in pairSpec.values())
            pairCacheSort.add(tuple([targetPair, pairCount]))
    else: 
        for i in range(len(pairCacheSort)):
            candidateMaxPair = pairCacheSort[-1] # pop the candidate max pair from pairCacheSort
            if candidateMaxPair[0] in pairCache and candidateMaxPair[1] == sum(pairSpecMap[0] * pairSpecMap[1] for pairSpecMap in pairCache.get(candidateMaxPair[0]).values()): 
                maxPair, maxPairCount = candidateMaxPair
                break # stop if found a valid cache sort pair
            else:
                logger.debug(f"stale cache: {candidateMaxPair} -> remove")
                pairCacheSort.pop()
    return tuple([maxPair, maxPairCount]), pairCache, pairCacheSort

# def join

"""
    return new pre token after maxpair is merged
"""
def mergePretokenCache(pretokens, maxPair, pairCache, pairSortedCache: SortedList):
    # loop through all pretoken
    # replace pair of bytes with maxPair
    maxPairBytes = maxPair[0]
    wordsToProcess = pairCache[maxPairBytes]
    
    newPairsToSort = set()
    for wordChars in wordsToProcess: 
        # create the cache for max pair (maxword -> {word: count})
        newPretoken = []
        m = 0
        while m < len(wordChars): 
            if m < len(wordChars)-1 and tuple([wordChars[m], wordChars[m+1]]) == maxPair[0]: 
                newPretoken.append(''.join(maxPair[0]))
                # todo: reindex pair cache after merge
                m=m+2 # advance past current pair
            else:
                newPretoken.append(wordChars[m])
                m = m+1
        
        newPretoken = tuple(newPretoken)
        # iterate through pair in new Pretoken to do the recalculation
        logger.debug(f"merging word: {wordChars}, max pair: {maxPair}")
        currentWordIndex = 0
        newIndex = 0
        currentWordIndicesProcessed = set() 
        newIndicesProcessed = set() 
        while currentWordIndex < (len(wordChars) - 1):
            checkingPretokenPair = tuple([wordChars[currentWordIndex], wordChars[currentWordIndex+1]])
            logger.debug(f"checking pair {checkingPretokenPair} at index: {currentWordIndex} of word {wordChars} ")
            if checkingPretokenPair == maxPair[0]: 
                if currentWordIndex > 0:
                    newOverlappingPairBehind = tuple([newPretoken[newIndex-1], ''.join(maxPair[0])]) # h, ow -> only place need join
                    oldAdjacentPair = tuple([wordChars[currentWordIndex-1], wordChars[currentWordIndex]]) # h, o
                    checkNewIndex = newIndex-1
                    checkCurrentIndex = currentWordIndex-1

                    if (checkNewIndex not in newIndicesProcessed): 
                        if newOverlappingPairBehind not in pairCache: 
                            pairCache[newOverlappingPairBehind] = {}
                        if newPretoken not in pairCache[newOverlappingPairBehind]: 
                            pairCache[newOverlappingPairBehind][newPretoken] = [pairCache[oldAdjacentPair][wordChars][0], 0]
                        pairCache[newOverlappingPairBehind][newPretoken][1] = pairCache[newOverlappingPairBehind][newPretoken][1] + 1 # only append 1 to occurences.
                        newIndicesProcessed.add(checkNewIndex)
                        newPairsToSort.add(newOverlappingPairBehind)
                    else: 
                        logger.debug(f"oops: found processed overlapp pair behind at new index {checkNewIndex}, old pair {oldAdjacentPair}, new pair {newOverlappingPairBehind}")

                    if (checkCurrentIndex not in currentWordIndicesProcessed): 
                        pairCache[oldAdjacentPair][wordChars][1] = pairCache[oldAdjacentPair][wordChars][1] - 1 # subtract 1
                        if pairCache[oldAdjacentPair][wordChars][1] <= 0: 
                            logger.debug(f"old adjacent pair {oldAdjacentPair}, word {wordChars} reaches 0 occurences, removing from pair cache")
                            del pairCache[oldAdjacentPair][wordChars] # remove the old pair cache. Ex. {h, o -> {(h, o, w) -> 1, (h, o, t) -> 1}} => {h, o -> {(h, o, t) -> 1}}
                        if not pairCache[oldAdjacentPair]: 
                            logger.debug(f"old adjacent pair {oldAdjacentPair} has no relevant word left, removing from pair cache")
                            del pairCache[oldAdjacentPair]
                        currentWordIndicesProcessed.add(checkCurrentIndex)
                        newPairsToSort.add(oldAdjacentPair)
                    else: 
                        logger.debug(f"oops: found processed overlapp pair behind at old index {checkCurrentIndex}, old pair {oldAdjacentPair}, new pair {newOverlappingPairBehind}")

                if currentWordIndex < len(wordChars) - 2:
                    # newOverlappingPairAhead = tuple([''.join(maxPair[0]), ''.join(word[i+2])]) # ow, e
                    newOverlappingPairAhead = tuple([''.join(maxPair[0]), newPretoken[newIndex+1]]) # ow, e
                    checkNewIndex = newIndex
                    oldAdjacentPair = tuple([wordChars[currentWordIndex+1], wordChars[currentWordIndex+2]]) # w, e
                    checkCurrentIndex = currentWordIndex+1

                    if (checkNewIndex not in newIndicesProcessed): 
                        if newOverlappingPairAhead not in pairCache: 
                            pairCache[newOverlappingPairAhead] = {}
                        if newPretoken not in pairCache[newOverlappingPairAhead]: 
                            pairCache[newOverlappingPairAhead][newPretoken] = [pairCache[oldAdjacentPair][wordChars][0], 0]
                        pairCache[newOverlappingPairAhead][newPretoken][1] = pairCache[newOverlappingPairAhead][newPretoken][1] + 1 # only append 1 to occurences.
                        newIndicesProcessed.add(checkNewIndex)
                        newPairsToSort.add(newOverlappingPairAhead)
                    else: 
                        logger.debug(f"oops: found processed overlapp pair ahead at new index {checkNewIndex}, old pair {oldAdjacentPair}, new pair {newOverlappingPairAhead}")
                    if (checkCurrentIndex not in currentWordIndicesProcessed): 
                        pairCache[oldAdjacentPair][wordChars][1] = pairCache[oldAdjacentPair][wordChars][1] - 1 # subtract 1
                        if pairCache[oldAdjacentPair][wordChars][1] <= 0: 
                            logger.debug(f"old adjacent pair {oldAdjacentPair}, word {wordChars} reaches 0 occurences, removing from pair cache")
                            del pairCache[oldAdjacentPair][wordChars] # remove the old pair cache. Ex. {h, o -> {(h, o, w) -> 1, (h, o, t) -> 1}} => {h, o -> {(h, o, t) -> 1}}
                        if not pairCache[oldAdjacentPair]: 
                            logger.debug(f"old adjacent pair {oldAdjacentPair} has no relevant word left, removing from pair cache")
                            del pairCache[oldAdjacentPair]
                        currentWordIndicesProcessed.add(checkCurrentIndex)
                        newPairsToSort.add(oldAdjacentPair)
                    else: 
                        logger.debug(f"oops: found processed overlapp pair behind at old index {checkCurrentIndex}, old pair {oldAdjacentPair}, new pair {newOverlappingPairAhead}")    
                logger.debug(f"found max pair in new pretoken {newPretoken} at index {currentWordIndex}")
                currentWordIndex = currentWordIndex + 2
                newIndex = newIndex+1 # keep track of new index
                continue
            # only skip look ahead, look behind is never checked 
            # the count of this pair (non adjacent pair) is unchanged -> no need to resort this pair (count just moved from old word to new words)
            if (currentWordIndex >= len(wordChars) - 2 or (currentWordIndex < len(wordChars) - 2 and maxPair[0] != tuple([wordChars[currentWordIndex+1], wordChars[currentWordIndex+2]]))):
                logger.debug(f"found normal non overlapping pair at cur index {currentWordIndex}: {checkingPretokenPair}")
                if maxPair[0] == tuple(['o', 'o']) and checkingPretokenPair == tuple(['O','o']): 
                    print("here 2")
                # pairCache[checkingPretokenPair][newPretoken] = pairCache[checkingPretokenPair][word]
                if newPretoken not in pairCache[checkingPretokenPair]: 
                    pairCache[checkingPretokenPair][newPretoken] = [pairCache[checkingPretokenPair][wordChars][0], 0]               
                pairCache[checkingPretokenPair][newPretoken][1] = pairCache[checkingPretokenPair][newPretoken][1] + 1               
                # subtract 1 from pair cache of current pair for old word
                pairCache[checkingPretokenPair][wordChars][1] = pairCache[checkingPretokenPair][wordChars][1] - 1 #
                if pairCache[checkingPretokenPair][wordChars][1] <= 0:
                    logger.debug(f"current un-adjacent {checkingPretokenPair} has no relevant with word {wordChars} left, removing from pair cache")
                    del pairCache[checkingPretokenPair][wordChars]
            currentWordIndex = currentWordIndex + 1
            newIndex = newIndex+1 # keep track of new index of new Pre token
        pretokens[newPretoken] = pretokens[wordChars]         # add new preToken to pretokenMap
        del pretokens[wordChars] # remove word h, o, w
    
    if len(pairSortedCache) > 0: 
        pairSortedCache.pop() # remove the current max
        # add pair counts to sorted
        for pairToSort in newPairsToSort: 
            if (pairToSort in pairCache): 
                pairCount = sum(pairSpecMap[0] * pairSpecMap[1] for pairSpecMap in pairCache.get(pairToSort).values())
                pairSortedCache.add(tuple([pairToSort, pairCount]))
    
    del pairCache[maxPair[0]] # remove the old pair (as they are merged to one token), o, w -> {}
    # remove maxpair as pair from the cache after merge
    return pretokens



def initVocabGPT(delimTokens): 
    vocab = {}
    for k in range(len(delimTokens)): 
        vocab[k] = delimTokens[k]
    # iterate through first 256 possible values
    for b in range(0,256): 
        if 33 <= int(b) <= 126 or 161 <= int(b) <= 172 or 174 <= int(b): 
            vocab[len(vocab)] = chr(b)
    i = 0
    for b in range(0,256): 
        if not (33 <= int(b) <= 126 or 161 <= int(b) <= 172 or 174 <= int(b)): 
            vocab[len(vocab)] = chr(256+i)
            i = i+1
    return vocab

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
    output_path = "",
    split_text_token = "<|endoftext|>",
    chunk_size_to_process = 1024 * 2014,
    get_max_by_cache = True,
    get_init_multi_process = True,
    process_count = 1,
    outputMergeJson = True,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    start_time = time.perf_counter()
    if get_init_multi_process: 
        pretokens = initPretokenMultiProcess(inputPath=input_path, splitTokens=split_text_token,specialTokens=special_tokens, chunkSizeToProcess=chunk_size_to_process, processCount=process_count)
    else:
        pretokens = initPretoken(inputPath=input_path, splitTokens=split_text_token,specialTokens=special_tokens, chunkSizeToProcess=chunk_size_to_process)
    end_time1 = time.perf_counter()
    elapsed_time1 = end_time1 - start_time
    logger.info(f"Init pretokens: Vocab size: {vocab_size}, Elapsed time: {elapsed_time1:.4f} seconds")
    logger.info(f"Init pretokens: {len(pretokens)}")
    # logger.debug(f"init pretoken: {pretokens}")
    delimTokens = special_tokens + [split_text_token]
    vocab = initVocabGPT(delimTokens=delimTokens)
    merges = []
    pairCache = {}
    pairCacheSort = SortedList([], key=itemgetter(1, 0)) # compare count first, then value for ties
    end_time2 = time.perf_counter()
    for iteration in tqdm(range(vocab_size-len(vocab)), desc="Training vocab"): 
        logger.info(f"iteration {iteration}")
        maxPair, pairCache, pairCacheSort = getMaxPairCountCache2(pretokens, iteration == 0, pairCache, pairCacheSort)
        if (maxPair[1] > 0):
            logger.info(f"iter {iteration}: max pair: {maxPair}")
            pretokens = mergePretokenCache(pretokens, maxPair, pairCache, pairCacheSort)
            logger.info(f"iter {iteration}: pretokens size: {len(pretokens)}, pair cache size: {len(pairCache)}")
            # word = word.replace(chr(0x20), chr(288))
                    # word = word.replace(chr(0x0A), chr(266))
            vocab[len(vocab)] = ''.join(maxPair[0])
            merges.append(maxPair[0])
            logger.debug(f"merges: {merges}")
        else: 
            logger.info(f"iteration {iteration}: max pair not found, stop!")
            break
    #add special token to vocab
 
    end_time3 = time.perf_counter()
    elapsed_time2 = end_time3 - end_time2
    elapsed_time3 = end_time3 - start_time
    logger.warning(f"Vocab size: {vocab_size}, Init pretoken time: {elapsed_time1:.1f} seconds")
    logger.warning(f"Merge Elapsed time: {elapsed_time2:.1f} seconds")
    logger.warning(f"Total Elapsed time: {elapsed_time3:.1f} seconds")
    now = datetime.now()
    string_format = now.strftime("%y%m%d%H%M%S")
    logger.info(f"vocab: {vocab}")
    
    if output_path: 
        # Write vocab file
        fileNameVocab = f"{output_path}-bytes-c{get_max_by_cache}-{get_init_multi_process*process_count}-{vocab_size}-{string_format}-{elapsed_time3:.1f}-vocab.json"        
        serializeVocab = {}
        for key,val in vocab.items(): 
            serializeVocab[key] = val
            # file.write(f"{visual}:{key}\n")
        with open(fileNameVocab, "w") as f:
            json.dump(serializeVocab, f, indent=4)
        
        # Write merge file
        fileNameMerges = f"{output_path}-bytes-c{get_max_by_cache}-{get_init_multi_process*process_count}-{vocab_size}-{string_format}-{elapsed_time3:.1f}-merges.json"
        serializeMerges = []
        if outputMergeJson:
            for merge in merges: 
                serializeMerges.append(tuple([pair for pair in merge]))
            with open(fileNameMerges, "w") as f:
                json.dump(serializeMerges, f, indent=4)
        else: 
            with open(fileNameMerges, "w") as f:
                for merge in merges: 
                    serializeMerges.append(f"{merge[0]} {merge[1]}\n")
                f.writelines(serializeMerges)
    return vocab, merges