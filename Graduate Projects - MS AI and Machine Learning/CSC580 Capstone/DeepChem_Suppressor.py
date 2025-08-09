import sys
import io
import warnings

class SuppressOutput:
    """Context manager to suppress stdout and stderr output"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, *args):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def import_deepchem():
    """Import deepchem with suppressed warnings"""
    with SuppressOutput():
        import deepchem as dc
    return dc

def load_tox21_quiet():
    """Load Tox21 dataset with suppressed output"""
    with SuppressOutput():
        import deepchem as dc
        _, (train, valid, test), _ = dc.molnet.load_tox21()
    return train, valid, test

# Suppress additional warnings
warnings.filterwarnings("ignore", message=".*PyTorch.*")
warnings.filterwarnings("ignore", message=".*No normalization.*")
warnings.filterwarnings("ignore", message=".*Skipped loading.*")

print("DeepChem Suppressor loaded. Import messages will be suppressed.")