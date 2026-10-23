import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # policy_training/ -> r2r2r_dp
sys.path.insert(0, HERE)  # fixtures
