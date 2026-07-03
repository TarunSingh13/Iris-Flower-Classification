"""
Configuration File
------------------
Stores all project constants in one place.
"""

# Model Configuration
N_NEIGHBORS = 5

# Dataset Configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Paths
MODEL_PATH = "models/iris_model.pkl"
OUTPUT_FOLDER = "outputs"

# Graph Names
CONFUSION_MATRIX_FILE = "confusion_matrix.png"
ACCURACY_GRAPH_FILE = "accuracy_vs_k.png"

# K values for comparison
K_VALUES = [1, 3, 5, 7, 9, 11, 13, 15]