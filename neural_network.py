import numpy as np
from base_model import BaseLanguageModel


class SimpleNeuralNetwork(BaseLanguageModel):
    """
    A simple single-layer neural network for next word prediction.

    Architecture:
        Input (one-hot) -> Weight Matrix W -> Softmax -> Output (probabilities)

    This is the simplest neural approach to language modeling.
    Uses one-hot encoding where each word is a vector of size vocab_size
    with a single 1 at the word's index position.
    """

    def __init__(self, data_file="data.txt", learning_rate=0.1):
        """
        Initialize the simple neural network.

        Args:
            data_file: Path to training text file
            learning_rate: Step size for gradient descent
        """
        # Call parent constructor
        super().__init__(data_file, learning_rate)

        # Weight matrix: maps one-hot input to output distribution
        self.W = None

    def _create_training_pairs(self):
        """
        Create single-word input to next-word output pairs.

        For text "i love ai", creates:
            X: [idx("i"), idx("love")]
            Y: [idx("love"), idx("ai")]
        """
        self.X = []
        self.Y = []

        # Each word predicts the next word
        for i in range(len(self.words) - 1):
            self.X.append(self.word_to_index[self.words[i]])
            self.Y.append(self.word_to_index[self.words[i + 1]])

    def initialize(self):
        """
        Initialize weight matrix W.

        W has shape (vocab_size, vocab_size) because:
        - Input is one-hot vector of size vocab_size
        - Output is probability distribution of size vocab_size
        """
        # Small random values to break symmetry
        self.W = np.random.randn(self.vocab_size, self.vocab_size) * 0.01
        print(f"Initialized weight matrix: {self.W.shape}")

    def one_hot(self, index):
        """
        Convert word index to one-hot vector.

        Args:
            index: Integer index of the word

        Returns:
            Vector of zeros with a 1 at the index position

        Example:
            one_hot(2) with vocab_size=5 -> [0, 0, 1, 0, 0]
        """
        vec = np.zeros(self.vocab_size)
        vec[index] = 1
        return vec

    def forward(self, x_index):
        """
        Forward pass: one-hot input → probability distribution.

        Args:
            x_index: Index of input word

        Returns:
            Tuple of (predictions, one_hot_input)
        """
        # Convert index to one-hot vector
        x = self.one_hot(x_index)

        # Linear transformation: multiply by weights
        z = np.dot(x, self.W)

        # Apply softmax to get probabilities (inherited from base)
        pred = self.softmax(z)

        return pred, x

    def train(self, epochs=200, print_every=20):
        """
        Train using gradient descent.

        For each training pair:
        1. Forward pass to get predictions
        2. Compute loss
        3. Compute gradient
        4. Update weights
        """
        print(f"\nTraining for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0

            for x_idx, y_idx in zip(self.X, self.Y):
                # Convert target to one-hot for loss computation
                y = self.one_hot(y_idx)

                # Forward pass
                pred, x = self.forward(x_idx)

                # Compute loss (inherited from base)
                l = self.loss(pred, y)
                total_loss += l

                # Gradient: difference between prediction and target
                grad = pred - y

                # Update weights using outer product
                # dW = x^T * grad (gradient of loss w.r.t. weights)
                self.W -= self.learning_rate * np.outer(x, grad)

            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

        print("Training complete!")

    def predict_next_word(self, word):
        """
        Predict the most likely next word.

        Args:
            word: Input word as string

        Returns:
            Predicted next word as string
        """
        x_index = self.word_to_index[word]
        pred, _ = self.forward(x_index)
        predicted_index = np.argmax(pred)
        return self.index_to_word[predicted_index]

    def generate_sentence(self, start_word, length=5):
        """
        Generate text by repeatedly predicting next word.

        Args:
            start_word: Starting word for generation
            length: Number of words to generate

        Returns:
            Generated sentence as string
        """
        current_word = start_word
        sentence = [current_word]

        for _ in range(length):
            next_word = self.predict_next_word(current_word)
            sentence.append(next_word)
            current_word = next_word

        return " ".join(sentence)


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    model = SimpleNeuralNetwork(learning_rate=0.1)
    model.load_data()
    model.initialize()
    model.train(epochs=200, print_every=20)

    # ✅ SUCCESS: "i" → "love" (neural net learns the dominant pattern)
    print("\n✅ SUCCESS: Single word prediction works when pattern is clear")
    print(f"  'i' → '{model.predict_next_word('i')}'")
    print(f"  'cats' → '{model.predict_next_word('cats')}'")
    print(f"  Neural net learned: 'i' mostly leads to 'love' ✓")

    # ❌ FAILURE: "chase" → always same answer (can't see what came before)
    print("\n❌ FAILURE: Same word always gives same prediction regardless of context")
    pred1 = model.predict_next_word("chase")
    print(f"  'chase' → '{pred1}'  (always the same!)")
    print(f"  We NEED: 'cats chase' → 'dogs', 'dogs chase' → 'cats'")
    print(f"  But model only sees 'chase' — doesn't know who is chasing.")
    print(f"  Also: 'ai' and 'coding' are unrelated in one-hot space")
    print(f"  even though they behave identically in our data.")
    print(f"\n➡️  NEXT: Dense embeddings for word similarity → embeddings.py")
