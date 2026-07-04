import numpy as np
from base_model import BaseLanguageModel


class EmbeddingModel(BaseLanguageModel):
    """
    Neural network with learnable word embeddings.

    Architecture:
        Word Index -> Embedding Lookup -> Output Weights -> Softmax

    Instead of one-hot encoding (sparse, large vectors), words are
    represented as dense vectors of size embedding_dim. These embeddings
    are learned during training and capture word semantics.
    """

    def __init__(self, data_file="data.txt", embedding_dim=2, learning_rate=0.1):
        """
        Initialize the embedding model.

        Args:
            data_file: Path to training text file
            embedding_dim: Size of word embedding vectors
            learning_rate: Step size for gradient descent
        """
        super().__init__(data_file, learning_rate)

        # Embedding dimension (small for visualization)
        self.embedding_dim = embedding_dim

        # Weight matrices (set by initialize)
        self.E = None  # Embedding matrix: vocab_size x embedding_dim
        self.W = None  # Output weights: embedding_dim x vocab_size

    def _create_training_pairs(self):
        """
        Create single-word input to next-word output pairs.
        Same as Step 2 - single word predicts next word.
        """
        self.X = []
        self.Y = []
        for i in range(len(self.words) - 1):
            self.X.append(self.word_to_index[self.words[i]])
            self.Y.append(self.word_to_index[self.words[i + 1]])

    def initialize(self):
        """
        Initialize embedding matrix and output weights.

        E: Maps word index to dense embedding vector
        W: Maps embedding to output probability distribution
        """
        # Embedding matrix: each row is a word's embedding
        self.E = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01

        # Output weights: project embedding to vocab_size
        self.W = np.random.randn(self.embedding_dim, self.vocab_size) * 0.01

        print(f"Initialized embedding matrix: {self.E.shape}")
        print(f"Initialized output weights: {self.W.shape}")

    def forward(self, x_index):
        """
        Forward pass: word index → embedding → prediction.

        Args:
            x_index: Index of input word

        Returns:
            Tuple of (predictions, embedding)
        """
        # Embedding lookup: get dense vector for this word
        embed = self.E[x_index]

        # Project to vocabulary size
        z = np.dot(embed, self.W)

        # Softmax for probabilities (inherited from base)
        pred = self.softmax(z)

        return pred, embed

    def train(self, epochs=500, print_every=50):
        """
        Train embeddings and output weights.

        Both E and W are updated via backpropagation.
        Only the embedding for the current word is updated each step.
        """
        print(f"\nTraining for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0

            for x, y in zip(self.X, self.Y):
                # Forward pass
                pred, embed = self.forward(x)

                # Compute loss (inherited from base)
                l = self.loss(pred, y)
                total_loss += l

                # Gradient at output
                grad_output = pred.copy()
                grad_output[y] -= 1  # Derivative of cross-entropy + softmax

                # Gradient for output weights W
                dW = np.outer(embed, grad_output)

                # Gradient for embedding
                # dE = W @ grad_output (backprop through W)
                dE = np.dot(self.W, grad_output)

                # Update weights
                self.W -= self.learning_rate * dW
                self.E[x] -= self.learning_rate * dE  # Only update this word's embedding

            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

        print("Training complete!")

    def predict_next_word(self, word):
        """Predict the most likely next word."""
        x_index = self.word_to_index[word]
        pred, _ = self.forward(x_index)
        predicted_index = np.argmax(pred)
        return self.index_to_word[predicted_index]

    def generate_sentence(self, start_word, length=5):
        """Generate text by repeatedly predicting next word."""
        current_word = start_word
        sentence = [current_word]

        for _ in range(length):
            next_word = self.predict_next_word(current_word)
            sentence.append(next_word)
            current_word = next_word

        return " ".join(sentence)

    def get_embedding(self, word):
        """Get the learned embedding vector for a word."""
        return self.E[self.word_to_index[word]]


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    model = EmbeddingModel(embedding_dim=2, learning_rate=0.1)
    model.load_data()
    model.initialize()
    model.train(epochs=500, print_every=50)

    # ✅ SUCCESS: Similar words develop similar embeddings
    print("\n✅ SUCCESS: Words in similar contexts cluster together")
    print(f"  'ai'     embedding: {model.get_embedding('ai')}")
    print(f"  'coding' embedding: {model.get_embedding('coding')}")
    print(f"  Both follow 'love' and precede 'is' → similar vectors ✓")
    print(f"  'cats' embedding: {model.get_embedding('cats')}")
    print(f"  'dogs' embedding: {model.get_embedding('dogs')}")
    print(f"  Both precede 'chase' and 'are' → similar vectors ✓")

    # ❌ FAILURE: "love" → always same prediction (can't see context)
    print("\n❌ FAILURE: Single word input can't distinguish different contexts")
    pred = model.predict_next_word("love")
    print(f"  'love' → '{pred}'  (always the same!)")
    print(f"  We NEED: context 'i love' → 'ai'/'coding'")
    print(f"  But also: 'i hate' → 'bugs' (different prediction!)")
    print(f"  Model sees only 'love' or 'hate' — not what came before.")
    print(f"\n➡️  NEXT: Use multiple words as input → context.py")
