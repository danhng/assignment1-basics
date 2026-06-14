print(chr(0))
#a. character code point 0 return the null character
print(repr(chr(0)))
#b. the repr function return the printable representation of the object. 
#c. if string is not implemented, repr is used instead. 

""" Unicode2
What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than 
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various 
input strings?

1. UTF-8 uses fewer bytes per characters than UTF-16 and UTF-32 -> leading to fewer tokens per characters -> save computation during training
2. Null bytes pollutions if trained with UTF16, UTF32 -> Cause the model attention to dead bytes more. 
3. More packed in context windows 
"""

"""
Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string 
into a Unicode string. Why is this function incorrect? Provide an example of an input byte 
string that yields incorrect results.
>>> def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))

Answer: the function is Wrong because UTF8 characters are width-variables. Strings containing characters whose codepoint > 1 byte will fail. Ex: Ăbc 
"""

"""
Give a two-byte sequence that does not decode to any Unicode character(s)
Answer: 110xxxxx - 0xxxxxxx (Because 2 byte sequence always has to start with 110xxxx - 10xxxxxx)
"""


from operator import itemgetter

import numpy as np
import timeit

from sortedcontainers import SortedList

# # Setup data
# data_list = list(np.random.rand(10000000))
# data_array = np.array(data_list)

# # 1. Native Python For Loop
# def for_loop_max(data):
#     maximum = data[0]
#     for num in data:
#         if num > maximum:
#             maximum = num
#     return maximum

# # Benchmarks
# time_loop = timeit.timeit(lambda: for_loop_max(data_list), number=10)
# time_builtin = timeit.timeit(lambda: max(data_list), number=10)
# time_numpy = timeit.timeit(lambda: np.max(data_array), number=10)

# print(f"For Loop:     {time_loop:.4f} seconds")
# print(f"Built-in max: {time_builtin:.4f} seconds")
# print(f"NumPy max:    {time_numpy:.4f} seconds")

pairCacheSort = SortedList([tuple(["d", 2]), tuple(["b", 3]), tuple(["c", 2])], key=itemgetter(1, 0)) # compare count first, then value for ties
print(pairCacheSort)
print(pairCacheSort.pop())
print(pairCacheSort)









