import numpy as np
from base_model import BaseLanguageModel

class TransformerBlock(BaseLanguageModel):
    """
    Complete transformer block with all components integrated.

    Architecture:
        Tokens -> Embed + Pos -> MultiHeadAttn + Residual + Norm
              -> FFN + Residual + Norm -> Output (from last token)

    This is the decoder-only transformer architecture used in GPT models.
    Key features:
    - Multi-head self-attention with causal masking
    - Residual connections around each sublayer
    - Layer normalization after each sublayer
    - Position-wise feed-forward network
    """

    def __init__(self, data_file="data.txt", context_size=2, embedding_dim=2,
                 num_heads=2, learning_rate=0.1):
        super().__init__(data_file, learning_rate)

        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.ff_hidden_dim = embedding_dim * 2

        # Weights
        self.E = None
        self.P = None
        self.W = None
        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.Wo = None  # Output projection after attention
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def _create_training_pairs(self):
        self.X = []
        self.Y = []
        for i in range(len(self.words) - self.context_size):
            context = self.words[i:i + self.context_size]
            target = self.words[i + self.context_size]
            self.X.append([self.word_to_index[w] for w in context])
            self.Y.append(self.word_to_index[target])

    def initialize(self):
        """Initialize all weights for complete transformer."""
        self.E = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
        self.P = np.random.randn(self.context_size, self.embedding_dim) * 0.01
        self.W = np.random.randn(self.embedding_dim, self.vocab_size) * 0.01

        self.Wq = np.random.randn(self.num_heads, self.embedding_dim, self.head_dim) * 0.01
        self.Wk = np.random.randn(self.num_heads, self.embedding_dim, self.head_dim) * 0.01
        self.Wv = np.random.randn(self.num_heads, self.embedding_dim, self.head_dim) * 0.01
        self.Wo = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01

        self.W1 = np.random.randn(self.embedding_dim, self.ff_hidden_dim) * 0.01
        self.b1 = np.zeros(self.ff_hidden_dim)
        self.W2 = np.random.randn(self.ff_hidden_dim, self.embedding_dim) * 0.01
        self.b2 = np.zeros(self.embedding_dim)

        print(f"Initialized Complete Transformer Block")

    def layer_norm(self, x, eps=1e-5):
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def causal_mask(self, seq_len):
        """
        Create causal (lower triangular) mask.
        Position i can only attend to positions <= i.
        """
        return np.tril(np.ones((seq_len, seq_len)))

    def forward(self, x_indices):
        """Forward pass through complete transformer block."""
        # 1. Embedding + Position
        x = np.array([self.E[idx] + self.P[pos] for pos, idx in enumerate(x_indices)])
        seq_len = x.shape[0]

        # 2. Multi-head attention with causal mask
        mask = self.causal_mask(seq_len)
        head_outputs = []

        for h in range(self.num_heads):
            Q = x @ self.Wq[h]
            K = x @ self.Wk[h]
            V = x @ self.Wv[h]

            # Scaled dot-product attention
            scores = (Q @ K.T) / np.sqrt(self.head_dim)

            # Apply causal mask: set future positions to -inf
            scores = scores * mask - 1e9 * (1 - mask)

            weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            weights /= np.sum(weights, axis=1, keepdims=True)

            head_outputs.append(weights @ V)

        # Concatenate and project
        multi_head = np.concatenate(head_outputs, axis=1)
        context = multi_head @ self.Wo

        # 3. Residual + Norm
        x = self.layer_norm(x + context)

        # 4. Feed Forward (per token)
        hidden_all = []
        ffn_out = []
        for token in x:
            h = token @ self.W1 + self.b1
            h_relu = np.maximum(0, h)
            out = h_relu @ self.W2 + self.b2
            hidden_all.append(h_relu)
            ffn_out.append(out)

        ffn_out = np.array(ffn_out)
        hidden_all = np.array(hidden_all)

        # 5. Residual + Norm
        x = self.layer_norm(x + ffn_out)

        # 6. Last token prediction (autoregressive)
        last_token = x[-1]
        z = last_token @ self.W
        pred = self.softmax(z)

        return pred, x, hidden_all, last_token

    def train(self, epochs=500, print_every=50):
        """Train the transformer."""
        print(f"\nTraining for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0

            for x, y in zip(self.X, self.Y):
                pred, x_out, hidden_all, last_token = self.forward(x)
                total_loss += self.loss(pred, y)

                # Output gradient
                grad_output = pred.copy()
                grad_output[y] -= 1

                # Gradient for output weights
                dW = np.outer(last_token, grad_output)
                dFinal = self.W @ grad_output

                # Simplified backprop to embeddings
                dEmb = dFinal / len(x)
                for pos, idx in enumerate(x):
                    self.E[idx] -= self.learning_rate * dEmb
                    self.P[pos] -= self.learning_rate * dEmb

                # Update FFN (simplified)
                self.W1 -= self.learning_rate * 0.01 * self.W1
                self.W2 -= self.learning_rate * 0.01 * self.W2

                # Update attention (simplified)
                for h in range(self.num_heads):
                    self.Wq[h] -= self.learning_rate * 0.01 * self.Wq[h]
                    self.Wk[h] -= self.learning_rate * 0.01 * self.Wk[h]
                    self.Wv[h] -= self.learning_rate * 0.01 * self.Wv[h]

                self.W -= self.learning_rate * dW

            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

        print("Training complete!")

    def predict_next_word(self, context_words):
        x_indices = [self.word_to_index[w] for w in context_words]
        pred, _, _, _ = self.forward(x_indices)
        return self.index_to_word[np.argmax(pred)]

    def generate_sentence(self, start_words, length=5):
        context = start_words[:]
        sentence = context[:]
        for _ in range(length):
            next_word = self.predict_next_word(context)
            sentence.append(next_word)
            context = context[1:] + [next_word]
        return " ".join(sentence)


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    model = TransformerBlock(context_size=2, embedding_dim=2, num_heads=2, learning_rate=0.1)
    model.load_data()
    model.initialize()
    model.train(epochs=500, print_every=50)

    print("\n--- Predictions ---")
    for i in range(len(model.X)):
        context_words = [model.index_to_word[idx] for idx in model.X[i]]
        print(f"{context_words} → {model.index_to_word[model.Y[i]]}")

    print("\n--- Generated Sentence ---")
    print(model.generate_sentence(model.words[:model.context_size], length=5))
