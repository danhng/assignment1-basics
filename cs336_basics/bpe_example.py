
"""
Input: training corpus
Output: trained vocalbulary 
"""

import io

import logging

# 1. Create a custom logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# 2. Create handlers
c_handler = logging.StreamHandler()  # For console
f_handler = logging.FileHandler('app.log', mode='w')  # For file
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

# 3. Create formatters and add to handlers
c_format = logging.Formatter('%(levelname)s - line %(lineno)d - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(lineno)d - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# 4. Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.warning('This is a warning shown on the console')
logger.error('This is an error recorded in the file')

# return a tuples of (bytes) chars 
def tupleBytes(text): 
    return tuple(char.encode('utf-8') for char in text)

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
        for m in range(i, 0, -1):
            currentCandidate =  bytes(leftWord[:m], "utf-8")
            # currentCandidate =  leftWord[:m]
            # logger.debug(f"split - checking candidate {currentCandidate}")
            if currentCandidate in vocab: 
                output.append(currentCandidate)
                leftWord = leftWord[m:i] # update the candidate
                # logger.debug(f"split - add {currentCandidate} to output, left word: {leftWord}")
                break
            if m == 1: 
                logger.debug(f"split - error, could not find suitable vocab for {leftWord}, check the vocab again!")
                output.append(leftWord)
                leftWord = ""
                break
    logger.debug(f"split: {word} -> {output}")
    return output


def bpe_example(text, loopCount = 1): 
    # Step 1: Vocalbulary init
    # map(int -> bytes)
    vocab = {}
    for i in range (0,255): 
        b = i.to_bytes(1) 
        vocab[b] = i
    logger.debug(f"vocab: {vocab}")
    
    # Step 2: Pre training
    # dict[bytes, int] Ex. 
    pre_tokens = {}
    
    merges = []

    # iterate through all lines, trim by whitespaces
    linesIo = io.StringIO(text)
    for line in linesIo:
        line = line.strip()
        line_pretokens = line.split() # [low, low, lowest]
        for line_pretoken in line_pretokens: 
            line_pretoken_bytes = line_pretoken.encode('utf-8')
            logger.debug(type(line_pretoken_bytes[0]))
            # todo: update the pre token to reflect the new merges each round, output: list[tuple[bytes,...]]
            pre_tokens[line_pretoken_bytes] = pre_tokens.get(line_pretoken_bytes, 0) + 1
    logger.debug(f"pre token result: {pre_tokens}")
    
    
    # Step 3: BPE merge
    # run through loop until certain loop count reached
    for i in range(loopCount):
        # update the merges
        logger.debug("Round", i+1)
        pairFrequencyCount = {}
        pairChecked = set()
        maxFrequencyPairCount = 0
        maxPair = bytes()
        # iterate through each lines
        for line in io.StringIO(text):
            line = line.strip()
            words = line.split() # [low, low, lowest]
            for word in words: # low
                wordBytes = splitVocab(word, vocab) # tuple of bytes
                # check for adjacent pairs in each pre token
                pairs = list(zip(wordBytes[:-1], wordBytes[1:])) # TODO: based on current vocab
                logger.debug(f"Checking word {wordBytes} -> pairs {pairs}")
                
                for pair in pairs: 
                    pairBytes = b''.join(pair)
                    if pairBytes not in pairChecked: 
                        pairChecked.add(pairBytes)
                        # iterate through the pre token cache to calculate frequency
                        for key, value in pre_tokens.items():
                            logger.debug(f"checking pair {pairBytes} against vocab key: {key}")
                            
                            ### PROBLEM: WRONG, because the pair in vocab might not exists after each iteration. Ex. we in newest (round 1 found est -> newest becomes new-est -> 'we' no longer exists in new-est)
                            ### PROBLEM: IF THERE ARE MORE THAN 1 occurence of pair in each pre token word -> we need to take into account this
                            if pairBytes in key: 
                                pairFrequencyCount[pairBytes] = pairFrequencyCount.get(pairBytes, 0) + value
                                logger.debug(f"hit {pairBytes}, add {value} to final: {pairFrequencyCount[pairBytes]})")
                                # update the current max pair if frequency count is bigger or (the same count and lexicography bigger)
                                if (pairFrequencyCount[pairBytes] > maxFrequencyPairCount) or ((pairFrequencyCount[pairBytes] == maxFrequencyPairCount) and (pairBytes > maxPair)) : 
                                    maxPair = pairBytes
                                    maxFrequencyPairCount = pairFrequencyCount[pairBytes]
                                    logger.debug(f"update max pair: {pairBytes}, count: {maxFrequencyPairCount}")
            logger.debug(f"pair checked after {words}: {pairChecked}")    
            # after iterate, add max pair to vocab
        vocab[maxPair] = len(vocab)
        merges.append(maxPair)
            # todo: update the pretokens on new (vocab + merges) -> do we really need to update the token in the pre token? -> Fuck yes
        logger.debug(pairFrequencyCount)
        logger.debug(f"final max pair after round {i}: {maxPair}, count: {maxFrequencyPairCount}, new vocab: {merges}")


        

text = "low low low low low\r\nlower lower widest widest widest\r\nnewest newest newest newest newest newest"
bpe_example(text, loopCount=6)

    