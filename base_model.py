# =============================================================================
# base_model.py - Base Class for All Language Model Examples
# =============================================================================
# This module provides the minimal base class containing ONLY code that is
# common to ALL language model examples in this learning path.
#
# All model classes (Steps 2-13) inherit from BaseLanguageModel.
# =============================================================================

import numpy as np
from abc import ABC, abstractmethod


class BaseLanguageModel(ABC):
    """
    Abstract base class for all language model examples.

    Contains only the attributes and methods that are common to EVERY example:
    - Data loading and vocabulary building
    - Softmax activation function
    - Cross-entropy loss function

    Subclasses must implement:
    - _create_training_pairs(): How to create X, Y training data
    - initialize(): How to initialize model weights
    - forward(): The forward pass computation
    - train(): The training loop
    - predict_next_word(): How to predict the next word
    - generate_sentence(): How to generate text
    """

    def __init__(self, data_file="data.txt", learning_rate=0.1):
        """
        Initialize the base language model.

        Args:
            data_file: Path to the text file containing training data
            learning_rate: Learning rate for gradient descent optimization
        """
        # Core hyperparameters (common to all models)
        self.data_file = data_file
        self.learning_rate = learning_rate

        # Raw word list (set by load_data)
        self.words = None

        # Vocabulary attributes (set by load_data)
        self.vocab = None           # List of unique words, sorted
        self.vocab_size = None      # Number of unique words
        self.word_to_index = None   # Dict mapping word -> index
        self.index_to_word = None   # Dict mapping index -> word

        # Training data (set by _create_training_pairs)
        self.X = None  # Input data (format varies by model)
        self.Y = None  # Target data (format varies by model)

    # =========================================================================
    # Data Loading - Common vocabulary building logic
    # =========================================================================
    def load_data(self):
        """
        Load text data and build vocabulary.

        This method:
        1. Reads the text file and converts to lowercase
        2. Splits text into words
        3. Builds vocabulary (sorted unique words)
        4. Creates word <-> index mappings
        5. Calls _create_training_pairs() for model-specific data prep
        """
        # Read and tokenize the text file
        with open(self.data_file, "r") as f:
            self.words = f.read().lower().split()

        # Build vocabulary: sorted list of unique words
        # Sorting ensures consistent word-to-index mapping across runs
        self.vocab = sorted(list(set(self.words)))
        self.vocab_size = len(self.vocab)

        # Create bidirectional mappings between words and indices
        # word_to_index: "hello" -> 5
        # index_to_word: 5 -> "hello"
        self.word_to_index = {w: i for i, w in enumerate(self.vocab)}
        self.index_to_word = {i: w for w, i in self.word_to_index.items()}

        # Let subclass create training pairs (format varies by model)
        self._create_training_pairs()

        print(f"Loaded {len(self.words)} words, vocabulary size: {self.vocab_size}")
        print(f"Training pairs: {len(self.X)}")

    @abstractmethod
    def _create_training_pairs(self):
        """
        Create X (input) and Y (target) training pairs.

        Must be implemented by each subclass because:
        - Some models use single word input (Steps 2-3)
        - Some models use context windows (Steps 4-13)
        - Format of X varies (index vs list of indices)
        """
        pass

    # =========================================================================
    # Softmax - Identical implementation in ALL examples
    # =========================================================================
    @staticmethod
    def softmax(x):
        """
        Compute softmax activation function.

        Converts a vector of raw scores (logits) into probabilities.
        Uses the numerically stable version: subtract max before exp.

        Args:
            x: Input array of logits

        Returns:
            Array of probabilities that sum to 1

        Example:
            softmax([1, 2, 3]) ≈ [0.09, 0.24, 0.67]
        """
        # Subtract max for numerical stability (prevents overflow)
        # e^(x - max) is equivalent to e^x / e^max
        e = np.exp(x - np.max(x))

        # Normalize to get probabilities
        return e / np.sum(e)

    # =========================================================================
    # Loss Function - Cross-entropy loss (common to all)
    # =========================================================================
    def loss(self, pred, target):
        """
        Compute cross-entropy loss.

        Measures how well the predicted probability distribution matches
        the target. Lower loss = better prediction.

        Handles two target formats:
        - Integer index: target is the index of the correct word (Steps 3-13)
        - One-hot vector: target is a one-hot encoded array (Step 2)

        Args:
            pred: Predicted probability distribution over vocabulary
            target: Either an integer index or one-hot encoded array

        Returns:
            Scalar loss value (negative log probability of correct word)
        """
        if isinstance(target, (int, np.integer)):
            # Index-based target: -log(probability of correct word)
            # Adding 1e-9 prevents log(0) = -infinity
            return -np.log(pred[target] + 1e-9)
        else:
            # One-hot target: sum of -target * log(pred)
            # Only the correct class contributes (others are 0)
            return -np.sum(target * np.log(pred + 1e-9))

    # =========================================================================
    # Abstract Methods - Must be implemented by ALL subclasses
    # =========================================================================
    @abstractmethod
    def initialize(self):
        """
        Initialize model weights.

        Each model has different weight matrices depending on its architecture:
        - Step 2: Single weight matrix W
        - Step 3: Embedding E and output W
        - Steps 5+: Add attention weights Wq, Wk, Wv
        - Steps 7+: Add multi-head attention
        - Steps 8+: Add FFN weights W1, W2
        """
        pass

    @abstractmethod
    def forward(self, x):
        """
        Perform forward pass through the model.

        Takes input (word index or context indices) and produces
        probability distribution over vocabulary for next word.

        Args:
            x: Input data (format varies by model)

        Returns:
            Tuple containing at least the prediction probabilities
        """
        pass

    @abstractmethod
    def train(self, epochs=500, print_every=50):
        """
        Train the model using gradient descent.

        Each model implements its own training loop because:
        - Gradient computation varies by architecture
        - Different weight matrices need different updates

        Args:
            epochs: Number of training iterations over full dataset
            print_every: Print loss every N epochs
        """
        pass

    @abstractmethod
    def predict_next_word(self, input_data):
        """
        Predict the most likely next word.

        Args:
            input_data: Single word (str) or list of context words

        Returns:
            Predicted next word as string
        """
        pass

    @abstractmethod
    def generate_sentence(self, start_input, length=5):
        """
        Generate a sentence by repeatedly predicting next words.

        Args:
            start_input: Starting word(s) for generation
            length: Number of words to generate

        Returns:
            Generated sentence as string
        """
        pass

