# %%
with open("brown_nouns.txt", "r") as f:
    words = [line.strip().lower() for line in f if line.strip()]

print("Sample words:", words[:20])
print("Total words:", len(words))

# %%
class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0
        self.end_of_word = False

# %%
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            node.count += 1
        node.end_of_word = True

    def find_split_point(self, word):
        """
        Find the point where branching is maximum
        Return stem, suffix
        """
        node = self.root
        split_index = 0
        max_branching = 0

        for i, ch in enumerate(word):
            if ch not in node.children:
                break
            node = node.children[ch]
            branching = len(node.children)
            if branching > max_branching:
                max_branching = branching
                split_index = i + 1

        stem = word[:split_index]
        suffix = word[split_index:]
        return stem, suffix

# %%
prefix_trie = Trie()
for w in words:
    prefix_trie.insert(w)

print(prefix_trie.find_split_point("kites"))

# %%
suffix_trie = Trie()
for w in words:
    suffix_trie.insert(w[::-1])

def find_suffix_split(trie, word):
    """Find suffix using reversed trie"""
    node = trie.root
    split_index = 0
    max_branching = 0
    rev_word = word[::-1]

    for i, ch in enumerate(rev_word):
        if ch not in node.children:
            break
        node = node.children[ch]
        branching = len(node.children)
        if branching > max_branching:
            max_branching = branching
            split_index = i + 1

    suffix = rev_word[:split_index][::-1]
    stem = word[:-split_index] if split_index > 0 else word
    return stem, suffix

print(find_suffix_split(suffix_trie, "kites"))

# %%
results = []
for w in words[:50]:
    pre_stem, pre_suffix = prefix_trie.find_split_point(w)
    suf_stem, suf_suffix = find_suffix_split(suffix_trie, w)
    results.append((w, f"{pre_stem}+{pre_suffix}", f"{suf_stem}+{suf_suffix}"))

print("{:<15} {:<20} {:<20}".format("Word", "Prefix Trie", "Suffix Trie"))
print("-"*60)
for r in results:
    print("{:<15} {:<20} {:<20}".format(*r))

# %%
from collections import Counter

suffix_counter = Counter()
for w in words:
    _, suf = find_suffix_split(suffix_trie, w)
    if suf:
        suffix_counter[suf] += 1

print("Most common suffixes:")
for suf, cnt in suffix_counter.most_common(10):
    print(f"{suf}: {cnt}")

# %%
correct_prefix = 0
correct_suffix = 0

common_suffixes = {"s", "es", "ing", "ed"}

for w in words:
    _, pre_suf = prefix_trie.find_split_point(w)
    if pre_suf in common_suffixes:
        correct_prefix += 1

    _, suf_suf = find_suffix_split(suffix_trie, w)
    if suf_suf in common_suffixes:
        correct_suffix += 1

print("Prefix Trie correct splits:", correct_prefix)
print("Suffix Trie correct splits:", correct_suffix)

if correct_suffix > correct_prefix:
    print("=> Suffix trie works better for stemming in this dataset.")
else:
    print("=> Prefix trie works better for stemming in this dataset.")

# %%



