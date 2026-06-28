import os
import sys

# Make `import app...` work when running pytest from the vision-service dir.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
