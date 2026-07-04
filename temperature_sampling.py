import numpy as np
from base_model import BaseLanguageModel

class TransformerCausal(BaseLanguageModel):
    def __init__(self, context_size=3, embedding_dim=4, num_heads=2, learning_rate=0.1, data_file="data.txt"):
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
        self.Wo = None
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        # Data
        self.words = None
        self.vocab = None
        self.word_to_index = None
        self.index_to_word = None
        self.X = None
        self.Y = None

    def load_data(self):
        """Load and preprocess the data."""
        with open(self.data_file, "r") as f:
            self.words = f.read().lower().split()

        self.vocab = sorted(list(set(self.words)))
        self.word_to_index = {w: i for i, w in enumerate(self.vocab)}
        self.index_to_word = {i: w for w, i in self.word_to_index.items()}
        vocab_size = len(self.vocab)

        # Build dataset
        X, Y = [], []
        for i in range(len(self.words) - self.context_size):
            context = self.words[i:i + self.context_size]
            target = self.words[i + self.context_size]

            X.append([self.word_to_index[w] for w in context])
            Y.append(self.word_to_index[target])

        self.X = np.array(X)
        self.Y = np.array(Y)

        print(f"Data loaded: {len(self.words)} words, {len(self.vocab)} unique words.")

    def initialize_weights(self):
        """Initialize all the weights for the model."""
        vocab_size = len(self.vocab)

        self.E = np.random.randn(vocab_size, self.embedding_dim) * 0.01
        self.P = np.random.randn(self.context_size, self.embedding_dim) * 0.01
        self.W = np.random.randn(self.embedding_dim, vocab_size) * 0.01

        self.Wq = np.random.randn(self.num_heads, self.embedding_dim, self.head_dim) * 0.01
        self.Wk = np.random.randn(self.num_heads, self.embedding_dim, self.head_dim) * 0.01
        self.Wv = np.random.randn(self.num_heads, self.embedding_dim, self.head_dim) * 0.01
        self.Wo = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01

        self.W1 = np.random.randn(self.embedding_dim, self.ff_hidden_dim) * 0.01
        self.b1 = np.zeros(self.ff_hidden_dim)
        self.W2 = np.random.randn(self.ff_hidden_dim, self.embedding_dim) * 0.01
        self.b2 = np.zeros(self.embedding_dim)

        print("Weights initialized.")

    # -----------------------
    # Utils
    # -----------------------
    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def layer_norm(self, x, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def causal_mask(self, seq_len):
        # Lower triangular mask
        return np.tril(np.ones((seq_len, seq_len)))

    # -----------------------
    # Forward
    # -----------------------
    def forward(self, x_indices):
        """Forward pass for the transformer model."""
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

            # 🔥 APPLY CAUSAL MASK (correct way)
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

    # -----------------------
    # Loss
    # -----------------------
    def loss(self, pred, target):
        return -np.log(pred[target] + 1e-9)

    # -----------------------
    # Training
    # -----------------------
    def train(self, epochs=300, print_every=50):
        """Train the model."""
        print(f"\nTraining for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0

            for x, y in zip(self.X, self.Y):
                pred = self.forward(x)
                total_loss += self.loss(pred, y)

                grad = pred.copy()
                grad[y] -= 1

                # simple update (only output + embeddings for stability)
                last_token = np.array([self.E[idx] + self.P[pos] for pos, idx in enumerate(x)])[-1]

                dW = np.outer(last_token, grad)
                self.W -= self.learning_rate * dW

                dEmb = (self.W @ grad) / len(x)

                for pos, idx in enumerate(x):
                    self.E[idx] -= self.learning_rate * dEmb
                    self.P[pos] -= self.learning_rate * dEmb

            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

        print("Training complete!")

    # -----------------------
    # Prediction
    # -----------------------

    def sample_with_temperature(self, probs, temperature=1.0):
        """Sample from probability distribution with temperature."""
        # Apply temperature
        probs = np.log(probs + 1e-9) / temperature
        probs = np.exp(probs)
        probs /= np.sum(probs)

        return np.random.choice(len(probs), p=probs)

    def predict_next_word(self, context_words, temperature=1.0):
        """Predict next word with temperature sampling."""
        x = [self.word_to_index[w] for w in context_words]
        pred = self.forward(x)

        if temperature == 0:
            # Greedy (deterministic)
            next_idx = np.argmax(pred)
        else:
            # Temperature sampling
            next_idx = self.sample_with_temperature(pred, temperature)

        return self.index_to_word[next_idx]

    def generate_sentence(self, start_words, length=10, temperature=1.0):
        """Generate sentence with temperature control."""
        context = start_words[:]
        sentence = []

        for _ in range(length):
            next_word = self.predict_next_word(context, temperature)
            sentence.append(next_word)
            context = context[1:] + [next_word]

        return " ".join(start_words + sentence)


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    model = TransformerCausal(context_size=3, embedding_dim=4, num_heads=2, learning_rate=0.1)
    model.load_data()
    model.initialize_weights()
    model.train(epochs=300, print_every=50)

    print("\n--- Temperature Sampling Demo ---")
    start_words = model.words[:model.context_size]

    print(f"\nTemperature = 0.5 (more deterministic):")
    print(model.generate_sentence(start_words, length=10, temperature=0.5))

    print(f"\nTemperature = 1.0 (balanced):")
    print(model.generate_sentence(start_words, length=10, temperature=1.0))

    print(f"\nTemperature = 2.0 (more random/creative):")
    print(model.generate_sentence(start_words, length=10, temperature=2.0))

