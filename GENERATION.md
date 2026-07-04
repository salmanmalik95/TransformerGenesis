# Text Generation: Sampling Strategies (Optional Advanced Topics)

Once you've trained a transformer, you need to **generate** text (predict the next word repeatedly). These three steps control how the model generates text.

---

## Step 11: Causal Masking (Don't Look at the Future)

**The problem:** During training, the model can see *future* words when it shouldn't. If it learns to "cheat" by looking ahead, it won't work at generation time.

**The solution:** A **mask** (a filter) that blocks the model from seeing future words:

```
Causal Mask (lower triangular [triangle-shaped]):
Position:   0  1  2
Position 0: [1, 0, 0]  ← position 0 sees only itself
Position 1: [1, 1, 0]  ← position 1 sees 0 and itself
Position 2: [1, 1, 1]  ← position 2 sees all previous

Result: Position 0 can't cheat by looking at positions 1, 2
```

This makes the model **autoregressive** (generates one word at a time, using only previous words).

📄 [casual_masking.py](casual_masking.py)

---

## Step 12: Temperature Sampling (Control Creativity)

**The problem:** Always picking the highest-probability word produces repetitive output: "coding is fun fun fun..."

**The solution:** **Sample randomly** from the probabilities, but adjust the "sharpness" using temperature (a dial [control]):

```
Temperature Effect (for "coding is ___"):
              fun    hard   great   others
T=0.5:       90%    8%     1%      1%      ← repetitive, predictable
T=1.0:       45%    30%    18%     7%      ← balanced
T=2.0:       30%    25%    22%     23%     ← chaotic, anything possible
```

- **Low temperature:** Sharp peaks = boring, repetitive
- **High temperature:** Flat distribution = creative but risky

Combine with top-K sampling for best results.

📄 [temperature_sampling.py](temperature_sampling.py)

---

## Step 13: Top-K Sampling (Filter Nonsense)

**The problem:** Even with temperature, the model can pick weird words that don't make sense.

**The solution:** Only consider the **K most likely words**, ignore the rest:

```
All words by probability:
{fun: 45%, hard: 30%, great: 18%, bugs: 3%, chase: 2%, ...}
                                              ↓ filter these out

Top-K=3:
{fun: 48%, hard: 32%, great: 20%}   ← only sensible words

Then sample from these three
```

**Together (temperature + top-K):**
1. Adjust distribution sharpness with temperature
2. Filter to top K words
3. Randomly pick one

| Setting | Output | Quality |
|:--------|:------:|:--------:|
| T=0.5, K=2 | Always "fun" | Boring but correct |
| T=1.0, K=3 | "fun" or "hard" (varies) | Good balance |
| T=2.0, K=10 | Can be nonsense | Too risky |

Use **low temperature + small K for reliable output**, **higher temperature + larger K for creativity**.

📄 [k_sampling.py](k_sampling.py)

---

## Summary

These three steps control **text generation**:
- **Causal Masking:** Make sure the model can't cheat by looking at future words
- **Temperature:** Control how much randomness (creativity) you want
- **Top-K:** Filter out nonsensical predictions

You don't need to understand these to learn how transformers work—they come *after* training is complete.

**Time to understand:** 5-10 minutes (after you understand Steps 1-10)  
**When to use:** Only when generating text with a trained model
