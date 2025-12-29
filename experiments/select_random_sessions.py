import random
import sys
sys.path.insert(0, '.')
from capcan_validation import mapping

# Get all session IDs
sessions = list(mapping.keys())
# Select 10 random sessions
random.seed(42)  # For reproducibility
selected = random.sample(sessions, 10)
print(','.join(selected))
