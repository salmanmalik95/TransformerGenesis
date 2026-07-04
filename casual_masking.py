import numpy as np
from base_model import BaseLanguageModel

class CausalMaskingModel(BaseLanguageModel):
    """
    Transformer with causal masking for autoregressive generation.

    Causal masking ensures each position can only attend to PAST positions.
    This is essential for language models where we predict the next token
    without peeking at future tokens.

    Mask example for seq_len=3:
        [[1, 0, 0],   Position 0 sees only itself
         [1, 1, 0],   Position 1 sees positions 0, 1
         [1, 1, 1]]   Position 2 sees positions 0, 1, 2
    """

    def __init__(self, data_file="data.txt", context_size=3, embedding_dim=4,
                 num_heads=2, learning_rate=0.1):
        super().__init__(data_file, learning_rate)

        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.ff_hidden_dim = embedding_dim * 2

        self.E = None
        self.P = None
        self.W = None
        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.Wo = None
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

    def load_data(self):
        super().load_data()
        print(f"Context size: {self.context_size}")

    def initialize(self):
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

        print(f"Initialized with Causal Masking")

    def layer_norm(self, x, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def causal_mask(self, seq_len):
        """
        Create lower triangular causal mask.

        Returns matrix where mask[i,j] = 1 if i >= j (can attend), 0 otherwise.
        This prevents position i from attending to future positions j > i.
        """
        return np.tril(np.ones((seq_len, seq_len)))

    def forward(self, x_indices):
        """Forward pass with causal masking emphasized."""
        # 1. Embedding + Position
        x = np.array([self.E[idx] + self.P[pos] for pos, idx in enumerate(x_indices)])
        seq_len = x.shape[0]

        # 2. Multi-head attention WITH CAUSAL MASK
        mask = self.causal_mask(seq_len)
        head_outputs = []

        for h in range(self.num_heads):
            Q = x @ self.Wq[h]
            K = x @ self.Wk[h]
            V = x @ self.Wv[h]

            scores = (Q @ K.T) / np.sqrt(self.head_dim)

            # CAUSAL MASK: Set future positions to -infinity before softmax
            # mask * scores keeps valid positions, (1-mask) * -1e9 makes invalid -inf
            scores = scores * mask - 1e9 * (1 - mask)

            weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            weights /= np.sum(weights, axis=1, keepdims=True)

            head_outputs.append(weights @ V)

        multi_head = np.concatenate(head_outputs, axis=1)
        context = multi_head @ self.Wo

        # 3. Residual + Norm
        x = self.layer_norm(x + context)

        # 4. Feed Forward
        ffn_out = []
        for token in x:
            h = token @ self.W1 + self.b1
            h = np.maximum(0, h)
            out = h @ self.W2 + self.b2
            ffn_out.append(out)
        ffn_out = np.array(ffn_out)

        # 5. Residual + Norm
        x = self.layer_norm(x + ffn_out)

        # 6. Last token prediction
        last_token = x[-1]
        z = last_token @ self.W
        pred = self.softmax(z)

        return pred

    def train(self, epochs=300, print_every=50):
        print(f"\nTraining for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0

            for x, y in zip(self.X, self.Y):
                pred = self.forward(x)
                total_loss += self.loss(pred, y)

                grad = pred.copy()
                grad[y] -= 1

                last_token = np.array([self.E[idx] + self.P[pos]
                                       for pos, idx in enumerate(x)])[-1]

                dW = np.outer(last_token, grad)
                self.W -= self.learning_rate * dW

                dEmb = (self.W @ grad) / len(x)
                for pos, idx in enumerate(x):
                    self.E[idx] -= self.learning_rate * dEmb
                    self.P[pos] -= self.learning_rate * dEmb

            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

        print("Training complete!")

    def predict_next_word(self, context_words):
        x = [self.word_to_index[w] for w in context_words]
        pred = self.forward(x)
        return self.index_to_word[np.argmax(pred)]

    def generate_sentence(self, start_words, length=6):
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
    model = CausalMaskingModel(context_size=3, embedding_dim=4, num_heads=2, learning_rate=0.1)
    model.load_data()
    model.initialize()
    model.train(epochs=300, print_every=50)

    print("\n--- Generated Sentence ---")
    print(model.generate_sentence(model.words[:model.context_size], length=6))
