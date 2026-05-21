
"""
Input: training corpus
Output: trained vocalbulary 
"""

import io

def bpe_example(text, loopCount = 1): 
    # Step 1: Vocalbulary init
    # map(int -> bytes)
    vocab = {}
    for i in range (0,255): 
        b = i.to_bytes(1) 
        vocab[i] = b
    
    # Step 2: Pre training
    # dict[tuple[bytes, ...], int] Ex. 
    pre_tokens = {}
    
    # iterate through all lines, trim by whitespaces
    linesIo = io.StringIO(text)
    for line in linesIo:
        line = line.strip()
        line_pretokens = line.split() # [low, low, lowest]
        for line_pretoken in line_pretokens: 
            # line_pretoken_bytes = tuple(line_pretoken)
            pre_tokens[line_pretoken] = pre_tokens.get(line_pretoken, 0) + 1
    print(f"pre token result: {pre_tokens}")
    # Step 3: BPE merge
    
    # run through loop until certain loop count reached
    for i in range(loopCount):
        print("Round", loopCount)
        pairFrequencyCount = {}
        pairChecked = set()
        for line in io.StringIO(text):
            line = line.strip()
            line_pretokens = line.split() # [low, low, lowest]
            for line_pretoken in line_pretokens: 
                # check for adjacent pairs in each pre token
                pairs = list(zip(line_pretoken[:-1], line_pretoken[1:]))
                print(f"checking pre token {line_pretoken} -> pairs {pairs}")
                for pair in pairs: 
                    pair = "".join(pair)
                    if pair not in pairChecked : 
                        pairChecked.add(pair)
                        for key, value in pre_tokens.items():
                            # print(f"checking pair {pair} against {"".join(key)} ")
                            if pair in key: 
                                pairFrequencyCount[pair] = pairFrequencyCount.get(pair, 0) + value
                                # print(f"hit {pair}, add {value} to final: {pairFrequencyCount[pair]})")
    print(pairFrequencyCount)
                            
# def checkSubTuple(subTuple, Tuple): 
    