import numpy as np
from base_model import BaseLanguageModel

class TopKSamplingModel(BaseLanguageModel):
    """
    Transformer with Top-K sampling for controlled text generation.

    Top-K sampling:
    1. Get probability distribution over vocabulary
    2. Keep only the K highest probability tokens
    3. Re-normalize these K probabilities
    4. Sample from this filtered distribution

    Benefits:
    - Prevents sampling very unlikely tokens (reduces nonsense)
    - Still allows diversity (unlike greedy)
    - Can combine with temperature for fine control
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

        print(f"Initialized with Top-K Sampling")

    def layer_norm(self, x, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def causal_mask(self, seq_len):
        return np.tril(np.ones((seq_len, seq_len)))

    def forward(self, x_indices):
        """Forward pass through transformer."""
        x = np.array([self.E[idx] + self.P[pos] for pos, idx in enumerate(x_indices)])
        seq_len = x.shape[0]

        mask = self.causal_mask(seq_len)
        head_outputs = []

        for h in range(self.num_heads):
            Q = x @ self.Wq[h]
            K = x @ self.Wk[h]
            V = x @ self.Wv[h]

            scores = (Q @ K.T) / np.sqrt(self.head_dim)
            scores = scores * mask - 1e9 * (1 - mask)

            weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            weights /= np.sum(weights, axis=1, keepdims=True)

            head_outputs.append(weights @ V)

        multi_head = np.concatenate(head_outputs, axis=1)
        context = multi_head @ self.Wo

        x = self.layer_norm(x + context)

        ffn_out = []
        for token in x:
            h = token @ self.W1 + self.b1
            h = np.maximum(0, h)
            out = h @ self.W2 + self.b2
            ffn_out.append(out)
        ffn_out = np.array(ffn_out)

        x = self.layer_norm(x + ffn_out)

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

    def sample_top_k(self, probs, k=3):
        """
        Sample from top K candidates only.

        Args:
            probs: Full probability distribution
            k: Number of top candidates to consider

        Returns:
            Sampled index (will be one of top K)
        """
        # Get indices of top K probabilities
        indices = np.argsort(probs)[-k:]

        # Get the corresponding probabilities
        top_probs = probs[indices]

        # Re-normalize to sum to 1
        top_probs /= np.sum(top_probs)

        # Sample from top K
        return np.random.choice(indices, p=top_probs)

    def sample_with_temperature(self, probs, temperature=1.0):
        """Apply temperature to probabilities."""
        probs = np.log(probs + 1e-9) / temperature
        probs = np.exp(probs)
        probs /= np.sum(probs)
        return probs

    def predict_next_word(self, context_words, temperature=1.0, k=3):
        """Predict next word with Top-K sampling."""
        x = [self.word_to_index[w] for w in context_words]
        pred = self.forward(x)

        # Apply temperature first (if not 1.0)
        if temperature != 1.0:
            pred = self.sample_with_temperature(pred, temperature)

        # Then apply top-k filtering
        next_idx = self.sample_top_k(pred, k)

        return self.index_to_word[next_idx]

    def generate_sentence(self, start_words, length=10, temperature=1.0, k=3):
        """Generate sentence with Top-K sampling."""
        context = start_words[:]
        sentence = []

        for _ in range(length):
            next_word = self.predict_next_word(context, temperature, k)
            sentence.append(next_word)
            context = context[1:] + [next_word]

        return " ".join(start_words + sentence)


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    model = TopKSamplingModel(context_size=3, embedding_dim=4, num_heads=2, learning_rate=0.1)
    model.load_data()
    model.initialize()
    model.train(epochs=300, print_every=50)

    print("\n--- Top-K Sampling Demo ---")
    start_words = model.words[:model.context_size]

    print(f"\nTop-K=2, Temperature=0.5:")
    print(model.generate_sentence(start_words, length=10, temperature=0.5, k=2))

    print(f"\nTop-K=3, Temperature=1.0:")
    print(model.generate_sentence(start_words, length=10, temperature=1.0, k=3))

    print(f"\nTop-K=5, Temperature=1.5:")
    print(model.generate_sentence(start_words, length=10, temperature=1.5, k=5))

