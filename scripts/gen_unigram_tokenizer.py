from transformers import AutoTokenizer
from collections import defaultdict
from math import log

import csv

corpus_path = "datasets/train.csv"
tokenizer_output_path = "models/unigram_tokenizer.csv"

# for 100M model (the end goal), do 24576 tokens
# for current testing, do 4096 tokens
tokenizer_vocab_size = 4096
tokenizer_base_name = "xlnet/xlnet-base-cased"

special_tokens = {
    "<unk>": 1,  # unknown token
    "<pad>": 1,  # padding token
    "<bos>": 1,  # beginning of sentence
    "<eos>": 1,  # end of sentence
    "<sep>": 1,  # separator token (for sentence pairs)
    "<cls>": 1,  # classification token (for sentence pairs)
    "<toolcall>": 1,  # tool call token (for generation with tools)
    "</toolcall>": 1,  # tool call end token (for generation with tools)
    "<toolresponse>": 1,  # tool response token (for generation with tools)
    "</toolresponse>": 1,  # tool response end token (for generation with tools)
    "<think>": 1,  # thought token (for chain-of-thought generation)
    "</think>": 1,  # thought end token (for chain-of-thought generation)
    "<system>": 1,  # system message token (for chat)
    "</system>": 1,  # system message end token (for chat)
    "<user>": 1,  # user message token (for chat)
    "</user>": 1,  # user message end token (for chat)
    "<assistant>": 1,  # assistant message token (for chat)
    "</assistant>": 1,  # assistant message end token (for chat)
}

tokenizer = AutoTokenizer.from_pretrained(tokenizer_base_name)

word_freqs = defaultdict(int)

with open(corpus_path, "r", encoding="utf-8") as corpus:
    reader = csv.reader(corpus)
    next(reader)  # skip header
    for line in reader:
        tokens = tokenizer.tokenize(line[2])
        for token in tokens:
            word_freqs[token] += 1

char_freqs = defaultdict(int)
subword_freqs = defaultdict(int)

for word, freq in word_freqs.items():
    for i in range(len(word)):
        char_freqs[word[i]] += freq
        # Loop through the subwords of length at least 2
        for j in range(i + 2, len(word) + 1):
            subword_freqs[word[i:j]] += freq

sorted_subwords = sorted(subword_freqs.items(), key=lambda x: x[1], reverse=True)

print(f"top 10 subwords: {sorted_subwords[:10]}")

token_freqs = (
    list(char_freqs.items())
    + sorted_subwords[: tokenizer_vocab_size - len(char_freqs) - len(special_tokens)]
    + list(special_tokens.items())
)
token_freqs = {token: freq for token, freq in token_freqs}

total_sum = sum([freq for _, freq in token_freqs.items()])
model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}


def encode_word(word, model):
    best_segmentations = [{"start": 0, "score": 1}] + [
        {"start": None, "score": None} for _ in range(len(word))
    ]
    for start_idx in range(len(word)):
        # This should be properly filled by the previous steps of the loop
        best_score_at_start = best_segmentations[start_idx]["score"]
        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                # If we have found a better segmentation ending at end_idx, we update
                if (
                    best_segmentations[end_idx]["score"] is None
                    or best_segmentations[end_idx]["score"] > score
                ):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}

    segmentation = best_segmentations[-1]
    if segmentation["score"] is None:
        # We did not find a tokenization of the word -> unknown
        return ["<unk>"], None

    score = segmentation["score"]
    start = segmentation["start"]
    end = len(word)
    tokens = []
    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])
    return tokens, score


print(encode_word("_Hopefully", model))
print(encode_word("_This", model))


loss = 0
for word, freq in word_freqs.items():
    _, word_loss = encode_word(word, model)
    if word_loss is not None:
        loss += freq * word_loss


print(f"Initial loss: {loss}")

with open(tokenizer_output_path, "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["token", "score"])

    for token, score in model.items():
        writer.writerow([token, score])
