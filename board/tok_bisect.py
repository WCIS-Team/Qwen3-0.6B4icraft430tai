import json
from tokenizers import Tokenizer

def T(d, label):
    json.dump(d, open("/tmp/t.json", "w", encoding="utf-8"))
    try:
        t = Tokenizer.from_file("/tmp/t.json")
        t.no_padding(); t.no_truncation()
        print(label, "OK", t.encode("ab", add_special_tokens=False).ids)
    except Exception as e:
        print(label, "FAIL", str(e)[:90])

print("tokenizers", __import__("tokenizers").__version__)
T({"version": "1.0", "model": {"type": "BPE", "vocab": {"a": 1, "b": 2, "ab": 3}, "merges": [["a", "b"]]}}, "tiny+merge")
T({"version": "1.0", "model": {"type": "BPE", "vocab": {"a": 1, "b": 2, "ab": 3}, "merges": []}}, "tiny-nomerge")

# now the real Qwen3 json with minimal model
d = json.load(open("/root/qwen3/config/qwen3/tokenizer.json", encoding="utf-8"))
m = d["model"]
base = {"type": "BPE", "vocab": m["vocab"], "merges": m["merges"]}
for k in ["dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix", "fuse_unk", "byte_fallback"]:
    if k in m:
        v = dict(base); v[k] = m[k]
        dd = dict(d); dd["model"] = v
        T(dd, "real+" + k)
