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










