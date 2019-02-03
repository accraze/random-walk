import random

num_train_sets = 100
num_sequences = 10  # or episodes


def generate_training_sets(num_sequences=10, num_train_sets=100):
    """Generate training data. Returns list of sequences."""
    return [
        [build_random_sequence() for i in range(num_sequences)]
        for i in range(num_train_sets)
    ]


def build_random_sequence():
    """Build a random sequence of steps. Returns list."""
    states = [3]  # Start in center at "D"
    while states[-1] not in [0, 6]:
        states.append(states[-1] + _random_step())
    return states


def _random_step():
    """Go left or right randomly. Returns int."""
    return (1 if random.choice([True, False]) else -1)
