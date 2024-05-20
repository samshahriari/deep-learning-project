from minbpe import RegexTokenizer

TOTAL_TOKENS = 2048
FILE_PATH = "../train.txt"

tokenizer = RegexTokenizer()

with open(FILE_PATH, "r") as f:
    text = f.read()

tokenizer.train(text, TOTAL_TOKENS)
tokenizer.register_special_tokens({"<eos>": 2048})
tokenizer.save("BPE")

