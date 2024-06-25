from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']
__version__ = '2.3.1+cpu'
debug = False
cuda: Optional[str] = None
git_version = 'd44533f9d073df13895333e70b66f81c513c1889'
hip: Optional[str] = None
