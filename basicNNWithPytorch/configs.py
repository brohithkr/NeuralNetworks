import sys
import os

project_path = os.path.dirname(sys.prefix)

dataset_path = os.path.join(project_path, './dataset')

device = 'cuda'