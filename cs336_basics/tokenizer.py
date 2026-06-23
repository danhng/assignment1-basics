from collections.abc import Iterable
import fast_bpe_bytes as fastBpeBytes
import fast_bpe_string as fastBpeString
import logging

# 1. Create a custom logger
logger = logging.getLogger('tokenizer')
logger.setLevel(logging.DEBUG)

class FastTokenizer: 
    vocab_id_word = {}
    vocab_word_id = {}
    merges = []
    merges_rank_map = {}
    special_tokens = []
    
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
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None): 
        return
    
    """
    wordBytes: tuples of bytes
    """
    def _findHighestRank(self, wordBytes): 
        maxPair = tuple()
        maxRank = 0
        i = 0
        while i < len(wordBytes) - 1: 
            targetPair = tuple([wordBytes[i], wordBytes[i+1]])
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
    def _merge(self, oldWordBytes, maxPair): 
        newWordBytes = []
        m = 0
        while m < len(oldWordBytes): 
            if m < len(oldWordBytes)-1 and tuple([oldWordBytes[m], oldWordBytes[m+1]]) == maxPair[0]: 
                newWordBytes.append(b''.join(maxPair[0]))
                m=m+2 # advance past current pair
            else:
                newWordBytes.append(oldWordBytes[m])
                m = m+1
        newWordBytes = tuple(newWordBytes)
        return newWordBytes
    
    """
    Encode an input text into a sequence of token IDs
    """
    def encode(self, text: str) -> list[int]: 
        # 1. decode to bytes
        wordBytes = text.encode("utf-8")
        targetWordBytes = tuple(bytes([b]) for b in wordBytes)
        
        # 2. while no matching new pair exists, find and merge the highest ranked pair in the current bytes
        maxPair = self._findHighestRank(targetWordBytes)
        while maxPair[1] > 0: 
            targetWordBytes = self._merge(targetWordBytes, maxPair)
            maxPair = self._findHighestRank(targetWordBytes)
        # 3. decode final bytes and return
        encoded = [self.vocab_word_id[word] for word in targetWordBytes]
        return encoded
    
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