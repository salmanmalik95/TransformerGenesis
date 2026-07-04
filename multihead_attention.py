import numpy as np
from base_model import BaseLanguageModel

class MultiHeadAttentionModel(BaseLanguageModel):
    """
    Neural network with multi-head attention.

    Architecture:
        Embeddings + Position -> [Head1, Head2, ...] -> Concat -> Output

    Multiple attention heads allow the model to attend to different
    aspects of the input simultaneously:
    - Head 1 might focus on nearby words
    - Head 2 might focus on semantic similarity
    - etc.

    Each head has dimension: head_dim = embedding_dim / num_heads
    """

    def __init__(self, data_file="data.txt", context_size=2, embedding_dim=2,
                 num_heads=2, learning_rate=0.1):
        """
        Initialize the multi-head attention model.

        Args:
            data_file: Path to training text file
            context_size: Number of previous words to use as context
            embedding_dim: Size of word embedding vectors
            num_heads: Number of parallel attention heads
            learning_rate: Step size for gradient descent
        """
        super().__init__(data_file, learning_rate)

        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Each head operates on a portion of the embedding
        self.head_dim = embedding_dim // num_heads

        # Weight matrices
        self.E = None   # Embedding matrix
        self.P = None   # Positional encoding
        self.W = None   # Output weights
        self.Wq = None  # Query weights [num_heads, embed_dim, head_dim]
        self.Wk = None  # Key weights
        self.Wv = None  # Value weights

    def _create_training_pairs(self):
        """Create context window to next-word pairs."""
        self.X = []
        self.Y = []

        for i in range(len(self.words) - self.context_size):
            context = self.words[i:i + self.context_size]
            target = self.words[i + self.context_size]
            context_indices = [self.word_to_index[w] for w in context]
            target_index = self.word_to_index[target]
            self.X.append(context_indices)
            self.Y.append(target_index)

    def load_data(self):
        """Load data and print configuration."""
        super().load_data()
        print(f"Context size: {self.context_size}, Num heads: {self.num_heads}")

    def initialize(self):
        """Initialize weights for multi-head attention."""
        self.E = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
        self.P = np.random.randn(self.context_size, self.embedding_dim) * 0.01
        self.W = np.random.randn(self.embedding_dim, self.vocab_size) * 0.01

        # Per-head weights: each head has its own projection matrices
        # Shape: [num_heads, embedding_dim, head_dim]
        self.Wq = np.random.randn(self.num_heads, self.embedding_dim, self.head_dim) * 0.01
        self.Wk = np.random.randn(self.num_heads, self.embedding_dim, self.head_dim) * 0.01
        self.Wv = np.random.randn(self.num_heads, self.embedding_dim, self.head_dim) * 0.01

        print(f"Initialized embedding matrix: {self.E.shape}")
        print(f"Initialized {self.num_heads} attention heads, head_dim: {self.head_dim}")

    def softmax(self, x):
        """Softmax activation function."""
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def forward(self, x_indices):
        """
        Forward pass with multi-head attention.

        Args:
            x_indices: List of word indices in context window

        Returns:
            Tuple of (predictions, context_vector)
        """
        # Step 1: Embeddings + Positional Encoding
        embeds = []
        for pos, idx in enumerate(x_indices):
            embeds.append(self.E[idx] + self.P[pos])
        embeds = np.array(embeds)

        # Step 2: Multi-head attention
        head_outputs = []

        for h in range(self.num_heads):
            # Q, K, V for this head
            Q = np.dot(embeds, self.Wq[h])  # (context, head_dim)
            K = np.dot(embeds, self.Wk[h])
            V = np.dot(embeds, self.Wv[h])

            # Attention scores
            scores = np.dot(Q, K.T)

            # Softmax row-wise
            e = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            weights = e / np.sum(e, axis=1, keepdims=True)

            # Weighted sum of values
            attended = np.dot(weights, V)  # (context, head_dim)
            head_outputs.append(attended)

        # Step 3: Concatenate all head outputs
        # Shape: (context, num_heads * head_dim) = (context, embedding_dim)
        multi_head_output = np.concatenate(head_outputs, axis=1)

        # Step 4: Average across context positions
        context_vector = np.mean(multi_head_output, axis=0)

        # Step 5: Final prediction
        z = np.dot(context_vector, self.W)
        pred = self.softmax(z)

        return pred, context_vector

    def loss(self, pred, target_index):
        """Cross-entropy loss."""
        return -np.log(pred[target_index] + 1e-9)

    def train(self, epochs=500, print_every=50):
        """Train multi-head attention model."""
        print(f"\nTraining for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0

            for x, y in zip(self.X, self.Y):
                # Forward pass
                pred, context_vector = self.forward(x)
                l = self.loss(pred, y)
                total_loss += l

                # Gradient of output
                grad_output = pred.copy()
                grad_output[y] -= 1

                # Gradient for W
                dW = np.outer(context_vector, grad_output)

                # Gradient for context vector
                dContext = np.dot(self.W, grad_output)

                # Distribute to heads (simplified)
                dMultiHead = dContext / len(x)

                # Update embeddings + positional encoding
                for pos, idx in enumerate(x):
                    self.E[idx] -= self.learning_rate * dMultiHead
                    self.P[pos] -= self.learning_rate * dMultiHead

                # Update attention weights (simplified regularization)
                for h in range(self.num_heads):
                    self.Wq[h] -= self.learning_rate * 0.01 * self.Wq[h]
                    self.Wk[h] -= self.learning_rate * 0.01 * self.Wk[h]
                    self.Wv[h] -= self.learning_rate * 0.01 * self.Wv[h]

                # Update output weights
                self.W -= self.learning_rate * dW

            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

        print("Training complete!")

    def predict_next_word(self, context_words):
        """Predict next word given context."""
        x_indices = [self.word_to_index[w] for w in context_words]
        pred, _ = self.forward(x_indices)
        predicted_index = np.argmax(pred)
        return self.index_to_word[predicted_index]

    def generate_sentence(self, start_words, length=5):
        """Generate text using multi-head attention."""
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
    model = MultiHeadAttentionModel(context_size=2, embedding_dim=2, num_heads=2, learning_rate=0.1)
    model.load_data()
    model.initialize()
    model.train(epochs=500, print_every=50)

    # Predictions
    print("\n--- Predictions ---")
    for i in range(len(model.X)):
        context_words = [model.index_to_word[idx] for idx in model.X[i]]
        print(f"{context_words} → {model.index_to_word[model.Y[i]]}")

    # Generate sentence
    print("\n--- Generated Sentence ---")
    print(model.generate_sentence(model.words[:model.context_size], length=5))
