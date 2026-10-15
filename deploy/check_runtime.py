"""Fail the worker build early if key runtime imports or CPU setup are broken."""
import argparse
import importlib
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--imports-only', action='store_true')
parser.parse_args()
for module in ['numpy', 'pandas', 'openpyxl', 'scipy', 'sklearn', 'Bio', 'anarci', 'torch', 'torchvision', 'tensorflow', 'keras', 'pytorch_lightning', 'transformers', 'accelerate', 'Levenshtein', 'pyteomics', 'easydict', 'wget']:
    importlib.import_module(module)
    print('OK', module, flush=True)
import torch
if torch.version.cuda is not None:
    raise SystemExit('Expected a CPU-only PyTorch wheel.')
for executable in ['ANARCI', 'hmmscan', 'conda']:
    if not shutil.which(executable):
        raise SystemExit(f'Missing executable: {executable}')
subprocess.run([
    'conda', 'run', '-n', 'TCR-PEP_Platform_oldversion', 'python', '-c',
    'import numpy, pandas, scipy, sklearn, imblearn; '
    'from imblearn.under_sampling import RandomUnderSampler; '
    'print("OK legacy epiTCR imports")',
], check=True)
print('CPU runtime imports passed. This is not an end-to-end prediction test.')
