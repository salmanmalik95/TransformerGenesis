# =============================================================================
# Step 1: Foundations - Tokenization, Training Pairs & Simple Bigram LLM
# =============================================================================
# This file combines three fundamental concepts:
# 1. Tokenization: Converting text to numbers
# 2. Training Pairs: Creating input-output pairs for learning
# 3. Bigram Language Model: A simple statistical approach to predict next word
# =============================================================================

from collections import defaultdict
import random

# =============================================================================
# PART 1: TOKENIZATION
# =============================================================================
# What is Tokenization?
# - Computers don't understand text, they understand numbers
# - Tokenization converts words/text into numerical representations
# - Each unique word gets a unique number (index)
# =============================================================================

print("=" * 60)
print("PART 1: TOKENIZATION")
print("=" * 60)

# Read text file
with open("data.txt", "r") as f:
    text = f.read().lower()

print("\n📄 Raw Text:")
print(text)

# Split into words (simple tokenization)
words = text.split()
print("\n📝 Words (Tokens):")
print(words)

# Create vocabulary (unique words, sorted)
vocab = sorted(list(set(words)))
print("\n📚 Vocabulary (Unique Words):")
print(vocab)
print(f"Vocabulary Size: {len(vocab)}")

# Create mappings: word <-> index
# This is the "dictionary" that maps words to numbers and vice versa
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for word, i in word_to_index.items()}

print("\n🔢 Word to Index Mapping:")
for word, idx in word_to_index.items():
    print(f"  '{word}' → {idx}")

# Encode the entire text as numbers
encoded = [word_to_index[word] for word in words]
print("\n🔐 Encoded Text (as numbers):")
print(encoded)

# Decode back to text (to verify)
decoded = [index_to_word[idx] for idx in encoded]
print("\n🔓 Decoded Text (back to words):")
print(" ".join(decoded))

# =============================================================================
# PART 2: TRAINING PAIRS
# =============================================================================
# What are Training Pairs?
# - To train a model to predict the "next word", we need examples
# - Each example has: INPUT (current word) → OUTPUT (next word)
# - We slide through the text creating these pairs
#
# Example: "i love ai" becomes:
#   X: [i, love]      Y: [love, ai]
#   Pair 1: i → love
#   Pair 2: love → ai
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: TRAINING PAIRS")
print("=" * 60)

# Create training pairs
X = []  # Inputs (current word indices)
Y = []  # Targets (next word indices)

for i in range(len(words) - 1):
    current_word = words[i]
    next_word = words[i + 1]

    X.append(word_to_index[current_word])
    Y.append(word_to_index[next_word])

print("\n📊 Training Data:")
print(f"X (inputs):  {X}")
print(f"Y (targets): {Y}")

print("\n🎯 Training Pairs (what the model learns):")
for x, y in zip(X, Y):
    print(f"  '{index_to_word[x]}' → '{index_to_word[y]}'")

print(f"\nTotal Training Examples: {len(X)}")

# =============================================================================
# PART 3: SIMPLE BIGRAM LANGUAGE MODEL
# =============================================================================
# What is a Bigram Model?
# - "Bi" = two, "gram" = unit → looks at pairs of words
# - Counts how often each word follows another word
# - Predicts next word based on frequency (most common wins)
#
# This is the SIMPLEST form of a language model!
# No neural network, just counting statistics.
#
# Example: If "i" is followed by "love" 3 times and "am" 1 time,
#          the model predicts "love" after "i" (higher frequency)
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: BIGRAM LANGUAGE MODEL")
print("=" * 60)

# Build bigram counts
# Structure: bigram_counts[current_word][next_word] = count
bigram_counts = defaultdict(lambda: defaultdict(int))

for i in range(len(words) - 1):
    current_word = words[i]
    next_word = words[i + 1]
    bigram_counts[current_word][next_word] += 1

print("\n📈 Bigram Counts (word → {next_word: count}):")
for word, next_words in bigram_counts.items():
    print(f"  '{word}' → {dict(next_words)}")


# Prediction function: Pick most frequent next word
def predict_next(word):
    """Predict the most likely next word based on bigram counts."""
    if word not in bigram_counts:
        return "<UNK>"  # Unknown word

    next_words = bigram_counts[word]
    predicted = max(next_words, key=next_words.get)
    return predicted


# Prediction function with randomness (weighted sampling)
def predict_next_random(word):
    """Predict next word with probability proportional to frequency."""
    if word not in bigram_counts:
        return "<UNK>"

    next_words = bigram_counts[word]
    words_list = list(next_words.keys())
    weights = list(next_words.values())

    return random.choices(words_list, weights=weights)[0]


# Text generation function
def generate_sentence(start_word, length=5, random_mode=False):
    """Generate text by repeatedly predicting next word."""
    result = [start_word]
    current = start_word

    for _ in range(length):
        if random_mode:
            next_word = predict_next_random(current)
        else:
            next_word = predict_next(current)

        if next_word == "<UNK>":
            break

        result.append(next_word)
        current = next_word

    return " ".join(result)


# =============================================================================
# ✅ SUCCESS: "i" → "love" (clear winner, count=2)
# =============================================================================
print("\n✅ SUCCESS: Bigram predicts well when one word dominates")
print(f"  'i' → '{predict_next('i')}'")
print(f"  (bigram_counts['i'] = {dict(bigram_counts['i'])})")
print(f"  'love' has count 2, 'hate' has count 1 → picks 'love' ✓")

# =============================================================================
# ❌ FAILURE: "chase" → ? (tied between "dogs" and "cats", can't decide)
# =============================================================================
print("\n❌ FAILURE: Bigram fails when counts are tied or context matters")
print(f"  'chase' → '{predict_next('chase')}'")
print(f"  (bigram_counts['chase'] = {dict(bigram_counts['chase'])})")
print(f"  'dogs' and 'cats' both have count 1 → picks arbitrarily!")
print(f"  We NEED: 'cats chase' → 'dogs', 'dogs chase' → 'cats'")
print(f"  But bigram only sees ONE word — can't learn from context.")
print(f"\n➡️  NEXT: A neural network that can LEARN patterns → neural_network.py")

