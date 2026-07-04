import numpy as np
from base_model import BaseLanguageModel

class PositionalEncodingModel(BaseLanguageModel):
    """
    Neural network with positional encoding added to embeddings.
    Allows the model to understand word order in the sequence.
    """

    def __init__(self, data_file="data.txt", context_size=2, embedding_dim=2, learning_rate=0.1):
        """
        Initialize the positional encoding model.

        Args:
            data_file: Path to training text file
            context_size: Number of previous words to use as context
            embedding_dim: Size of word embedding vectors
            learning_rate: Step size for gradient descent
        """
        super().__init__(data_file, learning_rate)

        self.context_size = context_size
        self.embedding_dim = embedding_dim

        # Weight matrices
        self.E = None   # Word embedding matrix
        self.P = None   # Positional encoding matrix (NEW)
        self.W = None   # Output weights
        self.Wq = None  # Query projection
        self.Wk = None  # Key projection
        self.Wv = None  # Value projection

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
        """Load data and print context size info."""
        super().load_data()
        print(f"Context size: {self.context_size}")

    def initialize(self):
        """Initialize embeddings, positional encoding, and attention weights."""
        self.E = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01

        # Positional encoding: one embedding per position in context
        self.P = np.random.randn(self.context_size, self.embedding_dim) * 0.01

        self.W = np.random.randn(self.embedding_dim, self.vocab_size) * 0.01
        self.Wq = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01
        self.Wk = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01
        self.Wv = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01

        print(f"Initialized embedding matrix: {self.E.shape}")
        print(f"Initialized positional encoding: {self.P.shape}")

    def softmax_rows(self, x):
        """Row-wise softmax for attention scores."""
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, x_indices):
        """
        Forward pass with positional encoding and attention.

        Args:
            x_indices: List of word indices in context window

        Returns:
            Tuple of (predictions, context_vector, attention_weights)
        """
        # Step 1: Embeddings + Positional Encoding
        # Each word gets its embedding PLUS its position embedding
        embeds = []
        for pos, idx in enumerate(x_indices):
            word_embed = self.E[idx]
            pos_embed = self.P[pos]
            embeds.append(word_embed + pos_embed)  # Element-wise addition
        embeds = np.array(embeds)

        # Step 2: Create Q, K, V
        Q = np.dot(embeds, self.Wq)
        K = np.dot(embeds, self.Wk)
        V = np.dot(embeds, self.Wv)

        # Step 3: Attention scores
        scores = np.dot(Q, K.T)

        # Step 4: Softmax (row-wise)
        weights = self.softmax_rows(scores)

        # Step 5: Weighted sum
        attended = np.dot(weights, V)

        # Step 6: Combine (simple average)
        context_vector = np.mean(attended, axis=0)

        # Step 7: Final prediction
        z = np.dot(context_vector, self.W)
        pred = self.softmax(z)

        return pred, context_vector, weights

    def train(self, epochs=500, print_every=50):
        """Train with positional encoding."""
        print(f"\nTraining for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0

            for x, y in zip(self.X, self.Y):
                pred, context_vector, weights = self.forward(x)
                l = self.loss(pred, y)
                total_loss += l

                # Gradient output
                grad_output = pred.copy()
                grad_output[y] -= 1

                # Gradient for W
                dW = np.outer(context_vector, grad_output)

                # Gradient for combined embedding
                dEmbed = np.dot(self.W, grad_output)

                # Distribute gradient to BOTH embeddings AND positional encoding
                for pos, idx in enumerate(x):
                    self.E[idx] -= self.learning_rate * (dEmbed / len(x))
                    self.P[pos] -= self.learning_rate * (dEmbed / len(x))  # Update position too!

                # Update W
                self.W -= self.learning_rate * dW

            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

        print("Training complete!")

    def predict_next_word(self, context_words):
        """Predict next word given context."""
        x_indices = [self.word_to_index[w] for w in context_words]
        pred, _, weights = self.forward(x_indices)
        predicted_index = np.argmax(pred)
        return self.index_to_word[predicted_index], weights

    def generate_sentence(self, start_words, length=5):
        """Generate text using positional encoding."""
        context = start_words[:]
        sentence = context[:]

        for _ in range(length):
            next_word, _ = self.predict_next_word(context)
            sentence.append(next_word)
            context = context[1:] + [next_word]

        return " ".join(sentence)


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    model = PositionalEncodingModel(context_size=2, embedding_dim=2, learning_rate=0.1)
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
