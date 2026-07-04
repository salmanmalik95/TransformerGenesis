# File Guide: Quick Reference

Complete reference of all Python files in TransformerGenesis. See what each file teaches and run them in order.

---

## 🧠 Part 1: Building the Transformer (Training & Architecture)

These files teach how transformers **learn** and understand language. Start here.

| Step | File | Purpose | Key Concept |
|:----:|------|---------|------------|
| 1 | [foundations.py](foundations.py) | Statistical word prediction (baseline) | Bigram counting (no learning) |
| 2 | [neural_network.py](neural_network.py) | Add learning ability to word prediction | Learnable weights |
| 3 | [embeddings.py](embeddings.py) | Compress word representation | Dense word vectors |
| 4 | [context.py](context.py) | Process multiple words at once | Context window |
| 5 | [attention.py](attention.py) | Focus on important words | Self-attention mechanism |
| 6 | [positional_encoding.py](positional_encoding.py) | Encode word position information | Position awareness |
| 7 | [multihead_attention.py](multihead_attention.py) | Multiple focus mechanisms in parallel | Multi-head attention |
| 8 | [ffn.py](ffn.py) | Learn complex patterns with hidden layer | Feed-forward + ReLU |
| 9 | [layer_norm.py](layer_norm.py) | Stabilize training | Layer normalization + residuals |
| 10 | [transformer_block.py](transformer_block.py) | Complete transformer architecture | Full transformer block |

**Time commitment:** 30-60 minutes (run each file, examine code)  
**Total parameters:** 106 (intentionally tiny for learning)  
**Code style:** Simple, under 150 lines per file

---

## 🎲 Part 2: Generating Text (Optional - Advanced Topics)

These files teach how to **use** a trained transformer to generate text. These are advanced/optional.

| Step | File | Purpose | Key Concept |
|:----:|------|---------|------------|
| 11 | [casual_masking.py](casual_masking.py) | Prevent looking at future words | Causal masking |
| 12 | [temperature_sampling.py](temperature_sampling.py) | Control randomness in predictions | Temperature control |
| 13 | [k_sampling.py](k_sampling.py) | Filter nonsensical predictions | Top-K filtering |

**Time commitment:** 5-10 minutes (after understanding Part 1)  
**When to use:** After training a model, during text generation  
**Prerequisite:** Understand Steps 1-10 first

📚 **Read [GENERATION.md](GENERATION.md) for detailed explanations of these steps**

---

## 📁 Utility Files

| File | Purpose |
|------|---------|
| [base_model.py](base_model.py) | Base class with shared functionality (vocabulary, softmax, loss) |
| [data.txt](data.txt) | Training dataset (12 sentences, 16 unique words) |
| [README.md](README.md) | Main guide with step-by-step explanations |
| [GENERATION.md](GENERATION.md) | Advanced guide for text generation strategies |

---

## 🚀 Quick Start

**For beginners:**
```bash
# Read the main guide first
# Then run the transformer files in order:
python foundations.py
python neural_network.py
python embeddings.py
python context.py
python attention.py
python positional_encoding.py
python multihead_attention.py
python ffn.py
python layer_norm.py
python transformer_block.py

# Optional: explore generation strategies
python casual_masking.py
python temperature_sampling.py
python k_sampling.py
```

**For experienced programmers:**
- Jump to [transformer_block.py](transformer_block.py) to see the complete model
- Use [layer_norm.py](layer_norm.py) as reference for normalization
- Check [ffn.py](ffn.py) for non-linearity patterns

---

## 📚 Learning Path

```
README.md (10-15 min)
         ↓
Files 1-10 (run in order, 30-60 min)
         ↓
GENERATION.md (5-10 min, optional)
         ↓
Files 11-13 (understand sampling, optional)
```

**Total time:** 30-60 minutes for complete understanding

---

## 🔗 File Dependencies

```
base_model.py (base class)
        ↓
All files inherit from BaseLanguageModel:
foundations.py → neural_network.py → embeddings.py → context.py
                                          ↓
                                     attention.py → positional_encoding.py
                                                         ↓
                                            multihead_attention.py
                                                         ↓
                                              ffn.py → layer_norm.py
                                                         ↓
                                            transformer_block.py
```

---

## 💡 What Each File Teaches

### Understanding Sequence
1. **foundations.py** - What's the baseline? (counting)
2. **neural_network.py** - How do we learn? (weights)
3. **embeddings.py** - How do we compress? (dense vectors)
4. **context.py** - How do we see context? (multiple words)
5. **attention.py** - How do we focus? (similarity scores)
6. **positional_encoding.py** - How do we track order? (position vectors)
7. **multihead_attention.py** - How do we specialize? (parallel heads)
8. **ffn.py** - How do we learn complexity? (non-linearity)
9. **layer_norm.py** - How do we stay stable? (normalization + residuals)
10. **transformer_block.py** - How does it all fit together? (complete model)

### Code Complexity
- **Files 1-3:** Basic ML patterns (lookup, matrix multiply, softmax)
- **Files 4-5:** Intermediate patterns (averaging, scoring)
- **Files 6-7:** Advanced patterns (learned transformations)
- **Files 8-9:** Expert patterns (ReLU, normalization, residuals)
- **File 10:** Full system assembly

---

## ✅ Checklist for Learning

- [ ] Read [README.md](README.md) intro (understand the big picture)
- [ ] Read Step 1 explanation in README
- [ ] Run [foundations.py](foundations.py) and read the code
- [ ] Repeat for Steps 2-10
- [ ] Run [transformer_block.py](transformer_block.py) with your own input
- [ ] (Optional) Read [GENERATION.md](GENERATION.md)
- [ ] (Optional) Run generation files (11-13)
- [ ] (Optional) Modify transformer_block.py (change embedding_dim, num_heads)

---

**Questions?** Check the relevant step in [README.md](README.md) or read the Python docstrings in the files!
