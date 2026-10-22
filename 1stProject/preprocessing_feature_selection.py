import sys
from pathlib import Path

sys.path.insert(0, "src")        # folder where eda_utils.py lives
import eda_utils as eu
from utils import preprocess     # if utils.py is inside src/, this works too

DATA_DIR = Path("data")

# load and link, same steps as in the EDA
trans = eu.load_transactions(DATA_DIR / "HI-Small_Trans.csv")
accounts = eu.load_accounts(DATA_DIR / "HI-Small_accounts.csv")
patterns, _, _ = eu.load_patterns(DATA_DIR / "HI-Small_Patterns.txt")
trans, _ = eu.link_accounts(trans, accounts)
trans, _, _ = eu.link_patterns(trans, patterns)

X, y, meta = preprocess(trans)
print(X.shape, int(y.sum()), "positives")
#display(X.head())