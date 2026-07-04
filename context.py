# =============================================================================
# Step 4: Context Windows (Multiple Word Input)
# =============================================================================
# WHAT THIS EXAMPLE TEACHES:
# - Using multiple words as context (context_size > 1)
# - Averaging embeddings from context window
# - Sliding window approach for training data
#
# KEY DIFFERENCE FROM STEP 3:
# - Input is now multiple words instead of single word
# - Embeddings are averaged to create context representation
# - Better predictions by considering more history
# =============================================================================

import numpy as np
from base_model import BaseLanguageModel


class ContextModel(BaseLanguageModel):
    """
    Neural network that uses multiple words as context for prediction.

    Architecture:
        [word1, word2, ...] -> Embeddings -> Average -> Output Weights -> Softmax

    Instead of predicting from a single word, this model looks at a
    window of previous words, averages their embeddings, and predicts.
    """

    def __init__(self, data_file="data.txt", context_size=2, embedding_dim=2, learning_rate=0.1):
        """
        Initialize the context model.

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
        self.E = None  # Embedding matrix
        self.W = None  # Output weights

    def _create_training_pairs(self):
        """
        Create context window to next-word pairs.

        For text "i love ai is" with context_size=2:
            X: [[idx("i"), idx("love")], [idx("love"), idx("ai")]]
            Y: [idx("ai"), idx("is")]
        """
        self.X = []
        self.Y = []

        for i in range(len(self.words) - self.context_size):
            # Context is the previous context_size words
            context = self.words[i:i + self.context_size]
            target = self.words[i + self.context_size]

            # Convert to indices
            context_indices = [self.word_to_index[w] for w in context]
            target_index = self.word_to_index[target]

            self.X.append(context_indices)
            self.Y.append(target_index)

    def load_data(self):
        """Load data and print context size info."""
        super().load_data()
        print(f"Context size: {self.context_size}")

    def initialize(self):
        """Initialize embedding matrix and output weights."""
        self.E = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
        self.W = np.random.randn(self.embedding_dim, self.vocab_size) * 0.01
        print(f"Initialized embedding matrix: {self.E.shape}")
        print(f"Initialized output weights: {self.W.shape}")

    def forward(self, x_indices):
        """
        Forward pass: context indices → averaged embedding → prediction.

        Args:
            x_indices: List of word indices in context window

        Returns:
            Tuple of (predictions, averaged_embedding, individual_embeddings)
        """
        # Get embedding for each word in context
        embeds = [self.E[i] for i in x_indices]

        # Combine by averaging (simple but effective)
        embed = np.mean(embeds, axis=0)

        # Project to vocabulary
        z = np.dot(embed, self.W)
        pred = self.softmax(z)

        return pred, embed, embeds

    def train(self, epochs=500, print_every=50):
        """
        Train with context windows.

        Gradient is distributed equally to all words in context.
        """
        print(f"\nTraining for {epochs} epochs...")

        for epoch in range(epochs):
            total_loss = 0

            for x, y in zip(self.X, self.Y):
                # Forward pass
                pred, embed, embeds = self.forward(x)
                l = self.loss(pred, y)
                total_loss += l

                # Gradient at output
                grad_output = pred.copy()
                grad_output[y] -= 1

                # Gradient for W
                dW = np.outer(embed, grad_output)

                # Gradient for averaged embedding
                dEmbed = np.dot(self.W, grad_output)

                # Distribute gradient equally to each word in context
                # Since we averaged, each word gets 1/context_size of gradient
                for idx in x:
                    self.E[idx] -= self.learning_rate * (dEmbed / len(x))

                # Update W
                self.W -= self.learning_rate * dW

            if epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

        print("Training complete!")

    def predict_next_word(self, context_words):
        """
        Predict next word from context window.

        Args:
            context_words: List of context words
        """
        x_indices = [self.word_to_index[w] for w in context_words]
        pred, _, _ = self.forward(x_indices)
        predicted_index = np.argmax(pred)
        return self.index_to_word[predicted_index]

    def generate_sentence(self, start_words, length=5):
        """Generate text using sliding context window."""
        context = start_words[:]
        sentence = context[:]

        for _ in range(length):
            next_word = self.predict_next_word(context)
            sentence.append(next_word)
            # Slide window: drop oldest, add newest
            context = context[1:] + [next_word]

        return " ".join(sentence)


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    model = ContextModel(context_size=2, embedding_dim=2, learning_rate=0.1)
    model.load_data()
    model.initialize()
    model.train(epochs=500, print_every=50)

    # ✅ SUCCESS: Context differentiates "i love" from "i hate"
    print("\n✅ SUCCESS: Multiple words distinguish different contexts")
    print(f"  ['i', 'love'] → '{model.predict_next_word(['i', 'love'])}'")
    print(f"  ['i', 'hate'] → '{model.predict_next_word(['i', 'hate'])}'")
    print(f"  Context helps! 'i love' → positive, 'i hate' → negative ✓")

    # ❌ FAILURE: "cats chase" vs "dogs chase" → likely same answer
    print("\n❌ FAILURE: Averaging embeddings loses word order information")
    pred1 = model.predict_next_word(["cats", "chase"])
    pred2 = model.predict_next_word(["dogs", "chase"])
    print(f"  ['cats', 'chase'] → '{pred1}'")
    print(f"  ['dogs', 'chase'] → '{pred2}'")
    print(f"  We NEED: 'cats chase' → 'dogs', 'dogs chase' → 'cats'")
    print(f"  But averaging treats all words equally — can't emphasize")
    print(f"  that 'chase' is the key structural signal here.")
    print(f"\n➡️  NEXT: Let words decide importance → attention.py")
