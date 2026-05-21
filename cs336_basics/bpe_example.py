
"""
Input: training corpus
Output: trained vocalbulary 
"""

import io


# return a tuples of (bytes) chars 
def tupleBytes(text): 
    return tuple(char.encode('utf-8') for char in text)

def bpe_example(text, loopCount = 1): 
    # Step 1: Vocalbulary init
    # map(int -> bytes)
    vocab = {}
    for i in range (0,255): 
        b = i.to_bytes(1) 
        vocab[i] = b
    
    # Step 2: Pre training
    # dict[bytes, int] Ex. 
    pre_tokens = {}
    
    merges = list[tuple[bytes, bytes]]

    # iterate through all lines, trim by whitespaces
    linesIo = io.StringIO(text)
    for line in linesIo:
        line = line.strip()
        line_pretokens = line.split() # [low, low, lowest]
        for line_pretoken in line_pretokens: 
            line_pretoken_bytes = line_pretoken.encode('utf-8')
            print(type(line_pretoken_bytes[0]))
            # todo: update the pre token to reflect the new merges each round, output: list[tuple[bytes,...]]
            pre_tokens[line_pretoken_bytes] = pre_tokens.get(line_pretoken_bytes, 0) + 1
    print(f"pre token result: {pre_tokens}")
    
    
    # Step 3: BPE merge
    # run through loop until certain loop count reached
    for i in range(loopCount):
        # update the merges
        print("Round", loopCount)
        pairFrequencyCount = {}
        pairChecked = set()
        for line in io.StringIO(text):
            line = line.strip()
            words = line.split() # [low, low, lowest]
            for word in words: # low
                wordBytes = tupleBytes(word) # tuple of bytes
                # check for adjacent pairs in each pre token
                pairs = list(zip(wordBytes[:-1], wordBytes[1:])) # TODO: based on current vocab
                print(f"checking word {wordBytes} -> pairs {pairs}")
                for pair in pairs: 
                    if pair not in pairChecked: 
                        pairBytes = b''.join(pair)
                        pairChecked.add(pairBytes)
                        for key, value in pre_tokens.items():
                            print(f"checking pair {pairBytes} against {key}")
                            if pairBytes in key: 
                                pairFrequencyCount[pairBytes] = pairFrequencyCount.get(pairBytes, 0) + value
                                print(f"hit {pairBytes}, add {value} to final: {pairFrequencyCount[pairBytes]})")
            # todo: update the pretokens on new (vocab + merges)
    print(pairFrequencyCount)

text = "tôi em"
bpe_example(text)

"""
Input: word: bytes to split, vocab: dict used to split
Output: list of bytes 
"""
def splitVocab(word, vocab): 
    # loop through all possible sub words from longest (whole word) to smallest (characters), 
    # if found in vocab -> add to output, remove the sub words from words. continue until no sub words left.
    # if not found in vocab after smallest, return the remaining sub words as is. 
    leftWord = word
    output = []
    while (len(leftWord) > 0):
        i = len(leftWord)
        while i < len(word): 
            currentCandidate =  bytes(word[:len(word) - len(leftWord)])
            if currentCandidate in vocab: 
                output.append(currentCandidate)
                leftWord = word[len(leftWord):]

        



    