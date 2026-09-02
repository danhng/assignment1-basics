from collections.abc import Iterable
import json
import os
import concurrent
import numpy as np
import regex as re
from tqdm import tqdm
from .fast_bpe_bytes import bytesToShiftedUnicode
from .fast_bpe_bytes import BASE_VOCAB_WORD_BYTE
import logging

# 1. Create a custom logger
logger = logging.getLogger('tokenizer')

VOCAB_MODE_BYTE = 1
VOCAB_MODE_UNICODE = 2

class FastTokenizer: 
    # vocab_id_word = {}
    # vocab_word_id = {}
    # merges = []
    # merges_rank_map = {}
    # special_tokens = []
    
    """
    vocab: dict[int, bytes]  
    merges: list[tuple[bytes, bytes]]  
    special_tokens: list[str] | None = None
    """
    def __init__(self, vocab, merges, special_tokens=None):
        # if vocab mode is byte -> 
        if isinstance(vocab[0], bytes):
            logger.debug("Vocab input is of bytes type -> convert to unicoded internal vocab")
            self.vocab_id_word = {key:bytesToShiftedUnicode(value, merge=True) for key, value in vocab.items()}
        else: 
            assert isinstance(vocab[0], str)
            logger.debug("Vocab input is of unicode type -> convert to unicoded internal vocab")
            self.vocab_id_word = vocab
        #word_id is simply the reverse of id_word dict
        
        self.vocab_word_id = {value:key for key, value in self.vocab_id_word.items()}
        
        if isinstance(merges[0][0], bytes): 
            logger.debug("merge input is of bytes type -> convert to unicoded internal vocab")
            self.merges = [tuple([bytesToShiftedUnicode(element, merge=True) for element in mergePair]) for mergePair in merges]
        else: 
            self.merges = merges
            
        if (special_tokens): 
            self.special_tokens = sorted(special_tokens, key=len, reverse=True) # todo: sort by order of length
        else: 
            self.special_tokens = special_tokens
        self.merges_rank_map = {merge: id for merge,id in zip(self.merges, range(len(self.merges), 0, -1))} # merge -> rank
        
        # get list of special token ids
        self.special_tokens_ids = {token: self.vocab_word_id[token] for token in special_tokens if token in self.vocab_word_id}
    
    def get_special_tokens_ids(self): 
        return self.special_tokens_ids
    
    def get_vocab_size(self): 
        return len(self.vocab_id_word)
    
    """
    Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the 
    same format that your BPE training code output) and (optionally) a list of special tokens. 
    This method should accept the following additional parameters:
        - vocab_filepath: str  
        - merges_filepath: str  
        - special_tokens: list[str] | None = None 
    """
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None, vocab_file_json=False): 
        # Open the file in read mode
        vocab = {}
        merges = []
        # deserialize vocab files
        with open(vocab_filepath, "r") as fileVocab:
            # Deserialize file content
            vocab_raw = json.load(fileVocab)
            vocab = {int(id): value for id, value in vocab_raw.items()}

        with open(merges_filepath, "r") as fileMerge:
            # Deserialize file content
            if vocab_file_json:
                mergesRaw = json.load(fileMerge)
                merges = [tuple(merge) for merge in mergesRaw]
            else:
                for line in fileMerge:
                # .strip() removes the newline character (\n) at the end of each line
                    mergesRaw = line.strip().split()
                    merges.append(tuple(mergesRaw))
        return cls(vocab, merges, special_tokens)
    
    #todo: find highest rank pairs belong to word. 
    
    """
    wordBytes: tuples of bytes
    """
    def _findHighestRank(self, transformedWordChars): 
        maxPair = tuple()
        maxRank = 0
        i = 0
        while i < len(transformedWordChars) - 1: 
            targetPair = tuple([transformedWordChars[i], transformedWordChars[i+1]])
            rank = self.merges_rank_map.get(targetPair, 0)
            logger.debug(f"Merge pair {targetPair} -> rank {rank}")
            if rank > 0 and rank > maxRank: 
                maxPair = targetPair
                maxRank = rank
                # i=i+2
                # break
            i=i+1             
        return maxPair, maxRank
    
    """
    wordBytes: tuples of bytes
    """
    def _merge(self, oldWordChars, maxPair): 
        newWordBytes = []
        m = 0
        while m < len(oldWordChars): 
            if m < len(oldWordChars)-1 and tuple([oldWordChars[m], oldWordChars[m+1]]) == maxPair[0]: 
                newWordBytes.append(''.join(maxPair[0]))
                m=m+2 # advance past current pair
            else:
                newWordBytes.append(oldWordChars[m])
                m = m+1
        newWordBytes = tuple(newWordBytes)
        return newWordBytes
    
    GPT2PretokenRegex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    """
    Encode an input text into a sequence of token IDs
    """
    def _encode(self, text: str) -> list[int]: 
        """
         splitTokenRegex = r"|".join(special_token for special_token in(self.special_tokens))
        matches = re.finditer(splitTokenRegex, text)
        output = []
        lastProcessedIndex = 0
        for match in matches: 
            specialTokenMatch = match.group()
            startSpecialTokenIndex, endSpecialTokenIndex = match.span()
            chunkText = text[lastProcessedIndex:startSpecialTokenIndex]
            encodedChunk = self._encode(chunkText)
        """
        # 1. encoded
        logger.debug(f"Chunk to encode: {text}")
        matches = re.finditer(self.GPT2PretokenRegex, text)
        output = []
        for match in matches: 
            wordBytes = match.group().encode("utf-8")
            transformedWordChars = bytesToShiftedUnicode(wordBytes)
            # logger.debug(f"{match} -> utf8 bytes: <{wordBytes}> -> transformed: <{transformedWordChars}>")
            # 2. while no matching new pair exists, find and merge the highest ranked pair in the current bytes
            maxPair = self._findHighestRank(transformedWordChars)
            while maxPair[1] > 0: 
                logger.debug(f"Find max pair: {maxPair}")
                transformedWordChars = self._merge(transformedWordChars, maxPair)
                maxPair = self._findHighestRank(transformedWordChars)
            # 3. decode final bytes and return
            encoded = [self.vocab_word_id[word] for word in transformedWordChars]
            output.extend(encoded)
            logger.debug(f"append {encoded}: current encoded chunk output: {output}")
        return output
    
    """
    Encode an input text into a sequence of token IDs
    """
    def encode(self, text: str) -> list[int]:         
        if self.special_tokens: 
            splitTokenRegex = r"|".join(re.escape(special_token) for special_token in(self.special_tokens))
            lastProcessedIndex = 0
            matches = re.finditer(splitTokenRegex, text)
            output = []
            for match in matches: 
                specialTokenMatch = match.group()
                startSpecialTokenIndex, endSpecialTokenIndex = match.span()
                chunkText = text[lastProcessedIndex:startSpecialTokenIndex]
                encodedChunk = self._encode(chunkText)
                output.extend(encodedChunk) # append text encoded
                output.append(self.vocab_word_id[specialTokenMatch]) # append the match
                lastProcessedIndex = endSpecialTokenIndex
                logger.debug(f"after adding {match.group()}: current encoded output: {output}")
            # if lastProcessedIndex == 0: 
            lastText = text[lastProcessedIndex:]
            encodedChunk = self._encode(lastText)
            output.extend(encodedChunk) # append text encoded
            logger.debug([self.vocab_id_word[encodedToken] for encodedToken in output])
            return output
        else: 
            logger.debug("No split token is provided -> adding whole text")
            encodedChunk = self._encode(text)
            return encodedChunk
    
    
    def read_text_chunks(self, filepath, batch_size):
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(batch_size)
                if not chunk:
                    break
                last_newline = chunk.rfind(b'\n')
                if last_newline == -1 and len(chunk) == batch_size:
                    # EDGE CASE: No newline found in the whole 64MB!
                    # f.readline() grabs the remainder of this exceptionally long line.
                    rest_of_line = f.readline()
                    valid_chunk = chunk + rest_of_line
                    
                elif last_newline != -1 and len(chunk) == batch_size:
                    # NORMAL CASE: Slice at the newline and rewind the leftovers
                    valid_chunk = chunk[:last_newline + 1]
                    leftover_bytes = len(chunk) - last_newline - 1
                    f.seek(-leftover_bytes, os.SEEK_CUR)
                    
                else:
                    # EOF CASE: Last chunk of the file
                    valid_chunk = chunk
                yield valid_chunk.decode('utf-8')
    
    def serialize_encode(self, input_path, output_filepath, batch_size_mb):
        # Pass 1: Count total tokens to pre-allocate memory map size
        total_tokens = 0
        batch_size_bytes = batch_size_mb * 1024 * 1024
        temp_bin_path = output_filepath + ".tmp.bin"
        # with open(input_path, 'r') as inputFile:
        batch = 0
        
        file_size = os.path.getsize(input_path)
        expected_batches = int(file_size / batch_size_bytes) + 1
        
        chunks = self.read_text_chunks(input_path, batch_size_bytes)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            with open(temp_bin_path, 'wb') as temp_file:
                for _id in tqdm(executor.map(self.encode, chunks), desc="Parallel Chunk Reading to count tokens"):
                    uint16_chunk = np.array(_id, dtype=np.uint16)
                    total_tokens += len(uint16_chunk)
                    batch = batch + 1
                    logger.info(f"Total tokens counted: {total_tokens}, batch {batch}/{expected_batches}")
                    temp_file.write(uint16_chunk.tobytes())
                
        # Pre-allocate a .npy-compatible raw memmap file on disk
        # Use np.lib.format.open_memmap instead of np.memmap
        # This creates a valid .npy file with the correct header
        mmap = np.lib.format.open_memmap(
            output_filepath, 
            mode='w+', 
            dtype=np.uint16, 
            shape=(total_tokens,)
        )
        # Pass 2: Write chunk by chunk into the memory map
        with open(temp_bin_path, 'rb') as temp_file:
            copy_chunk_bytes = 100 * 1024 * 1024
            offset = 0
            while True:
                raw_bytes = temp_file.read(copy_chunk_bytes)
                if not raw_bytes:
                    break
                arr = np.frombuffer(raw_bytes, dtype=np.uint16)
                mmap[offset : offset + len(arr)] = arr
                offset += len(arr)
                
        # Flush changes to disk
        mmap.flush()
        os.remove(temp_bin_path)
        logger.info("Serialization complete!")
        
    """
    -> Iterator[int] Given an iterable of 
    strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is 
    required for memory-efficient tokenization of large files that we cannot directly load into 
    memory.
    """
    def encode_iterable(self, iterable: Iterable[str]): 
        for text in iterable: 
            encoded = self.encode(text)
            yield encoded

    """
    -> str Decode a sequence of token IDs into text.
    """
    def decode(self, ids: list[int]): 
        # map token ids to chars 
        # map chars to bytes
        # encode utf 8
        output = ""
        bs = bytearray()
        
        # if we are given a nested list, flatten them first
        if (ids and isinstance(ids[0], list)): 
            ids = [item for sublist in ids for item in sublist]
        
        for id in ids:
            tokenWord = self.vocab_id_word[id]
            if  not self.special_tokens or (self.special_tokens and tokenWord not in self.special_tokens): # if no special token -> just append, if token is normal -> just append
                for c in tokenWord: # char -> decoded byte -> append to byte array
                    bs.append(BASE_VOCAB_WORD_BYTE[c])
            else: 
                if bs: 
                    output = output + bs.decode("utf-8") # flush byte array buffer up until special token
                    bs = bytearray()
                output = output + tokenWord # add special token
        if bs: 
                output = output + bs.decode("utf-8")
        return output