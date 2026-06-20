import tiktoken
gpt2_tokenizer = tiktoken.get_encoding("gpt2")

for i in range(33,127):
    print(f"{i}: {chr(i)}")

for i in range(161,324):
    print(f"{i}: {chr(i)}")