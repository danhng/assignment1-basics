from collections.abc import Iterable
import json
import regex as re
import fast_bpe_bytes as fastBpeBytes
import logging

# 1. Create a custom logger
logger = logging.getLogger('tokenizer')

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
        self.vocab_id_word = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab_word_id = {value:key for key, value in self.vocab_id_word.items()}
        self.merges_rank_map = {merge: id for merge,id in zip(merges, range(len(merges), 0, -1))} # merge -> rank
    
    """
    Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the 
    same format that your BPE training code output) and (optionally) a list of special tokens. 
    This method should accept the following additional parameters:
        - vocab_filepath: str  
        - merges_filepath: str  
        - special_tokens: list[str] | None = None 
    """
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None): 
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
            merges = json.load(fileMerge)
            mergesTuples = [tuple(merge) for merge in merges]
        return cls(vocab, mergesTuples, special_tokens)
    
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
            if rank > 0 and rank > maxRank: 
                maxPair = targetPair
                maxRank = rank
                i=i+2
                break
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
        # 1. encoded
        logger.debug(f"Word to encode: {text}")
        wordBytes = text.encode("utf-8")
        transformedWordChars = fastBpeBytes.bytesToShiftedUnicode(wordBytes)
        
        # 2. while no matching new pair exists, find and merge the highest ranked pair in the current bytes
        maxPair = self._findHighestRank(transformedWordChars)
        while maxPair[1] > 0: 
            logger.debug(f"Find max pair: {maxPair}")
            transformedWordChars = self._merge(transformedWordChars, maxPair)
            maxPair = self._findHighestRank(transformedWordChars)
        # 3. decode final bytes and return
        encoded = [self.vocab_word_id[word] for word in transformedWordChars]
        return encoded
    
    """
    Encode an input text into a sequence of token IDs
    """
    def encode(self, text: str) -> list[int]:         
        splitTokenRegex = r"|".join(re.escape(escapedToken) for escapedToken in(self.special_tokens))
        fullRegex = splitTokenRegex + r"|"+self.GPT2PretokenRegex
        matches = re.finditer(fullRegex, text)
        output = []
        for match in matches: 
            chunk = match.group()
            if chunk in self.special_tokens: 
                output.append(self.vocab_word_id[chunk])
            else: 
                encodedChunk = self._encode(chunk)
                output.extend(encodedChunk)
        return output
    
    """
    -> Iterator[int] Given an iterable of 
    strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is 
    required for memory-efficient tokenization of large files that we cannot directly load into 
    memory.
    """
    def encode_iterable(self, iterable: Iterable[str]): 
        return

    """
    -> str Decode a sequence of token IDs into text.
    """
    def decode(self, ids: list[int]): 
        return