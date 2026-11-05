from tokenizers import Tokenizer
t = Tokenizer.from_file('/root/qwen3/config/qwen3/tokenizer.json')
t.no_padding(); t.no_truncation()
cases = ["你好，世界", "Qwen3 是第三代", "2026 年 11 月 4 日", "Hello, world!", "abc def\nghi", "<|im_start|>user\nHi<|im_end|>", ""]
for s in cases:
    ids = t.encode(s, add_special_tokens=False).ids
    print(repr(s), "->", ids[:40])
print("ver", __import__("tokenizers").__version__)
