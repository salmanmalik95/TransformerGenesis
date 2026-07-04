import numpy as np
from base_model import BaseLanguageModel

class LayerNormModel(BaseLanguageModel):
    """
    Neural network with layer normalization and residual connections.
    These techniques stabilize training of deep networks.

    Architecture:
        x -> MultiHeadAttn -> Add(x) -> LayerNorm -> FFN -> Add -> LayerNorm -> Output

    Residual connections: Output = x + Sublayer(x)
    - Helps gradient flow in deep networks
    - Allows learning identity function easily

    Layer Norm: Normalize across features (not batch)
    - Stabilizes activations
    - Helps with training dynamics
    """

    def __init__(self, data_file="data.txt", context_size=2, embedding_dim=2, 
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
        
        self.W1 = np.random.randn(self.embedding_dim, self.ff_hidden_dim) * 0.01
        self.b1 = np.zeros(self.ff_hidden_dim)
        self.W2 = np.random.randn(self.ff_hidden_dim, self.embedding_dim) * 0.01
        self.b2 = np.zeros(self.embedding_dim)
        
        print(f"Initialized with Layer Normalization and Residual Connections")

    def layer_norm(self, x, eps=1e-5):
        """
        Layer normalization: normalize across features.

        For each position, normalize to zero mean and unit variance.
        """
        mean = np.mean(x)
        var = np.var(x)
        return (x - mean) / np.sqrt(var + eps)

    def feed_forward(self, x):
        """Two-layer FFN with ReLU."""
        h = np.dot(x, self.W1) + self.b1
        h = np.maximum(0, h)
        out = np.dot(h, self.W2) + self.b2
        return out, h

    def forward(self, x_indices):
        """Forward pass with residual connections and layer norm."""
        # Embeddings + Position
        embeds = []
        for pos, idx in enumerate(x_indices):
            embeds.append(self.E[idx] + self.P[pos])
        embeds = np.array(embeds)

        # Multi-head attention
        head_outputs = []
        for h in range(self.num_heads):
            Q = np.dot(embeds, self.Wq[h])
            K = np.dot(embeds, self.Wk[h])
            V = np.dot(embeds, self.Wv[h])

            scores = np.dot(Q, K.T)
            e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            weights = e / np.sum(e, axis=1, keepdims=True)
            attended = np.dot(weights, V)
            head_outputs.append(attended)

        multi_head_output = np.concatenate(head_outputs, axis=1)
        context_vector = np.mean(multi_head_output, axis=0)

        # Residual + LayerNorm after attention (NEW)
        context_vector = self.layer_norm(context_vector + np.mean(embeds, axis=0))

        # FFN
        ffn_out, hidden = self.feed_forward(context_vector)

        # Residual + LayerNorm after FFN (NEW)
        ffn_out = self.layer_norm(ffn_out + context_vector)

        # Final prediction
        z = np.dot(ffn_out, self.W)
        pred = self.softmax(z)

        return pred, context_vector, ffn_out, hidden

    def train(self, epochs=500, print_every=50):
        print(f"\nTraining for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0

            for x, y in zip(self.X, self.Y):
                pred, context_vector, ffn_out, hidden = self.forward(x)
                l = self.loss(pred, y)
                total_loss += l

                grad_output = pred.copy()
                grad_output[y] -= 1

                dW = np.outer(ffn_out, grad_output)
                dFFN = np.dot(self.W, grad_output)
                dW2 = np.outer(hidden, dFFN)
                db2 = dFFN
                dhidden = np.dot(self.W2, dFFN)
                dhidden[hidden <= 0] = 0
                dW1 = np.outer(context_vector, dhidden)
                db1 = dhidden
                dContext = np.dot(self.W1, dhidden)

                dMultiHead = dContext / len(x)
                for pos, idx in enumerate(x):
                    self.E[idx] -= self.learning_rate * dMultiHead
                    self.P[pos] -= self.learning_rate * dMultiHead

                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1
                self.W2 -= self.learning_rate * dW2
                self.b2 -= self.learning_rate * db2

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
    model = LayerNormModel(context_size=2, embedding_dim=2, num_heads=2, learning_rate=0.1)
    model.load_data()
    model.initialize()
    model.train(epochs=500, print_every=50)

    print("\n--- Predictions ---")
    for i in range(len(model.X)):
        context_words = [model.index_to_word[idx] for idx in model.X[i]]
        print(f"{context_words} → {model.index_to_word[model.Y[i]]}")

    print("\n--- Generated Sentence ---")
    print(model.generate_sentence(model.words[:model.context_size], length=5))

