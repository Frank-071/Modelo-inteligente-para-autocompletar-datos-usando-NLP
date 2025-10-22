# Carga y utilidades para CoNLL (TOKEN \t TAG)
from typing import List, Tuple

def read_conll(path: str) -> List[List[Tuple[str, str]]]:
    sents, sent = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if sent:
                    sents.append(sent); sent = []
                continue
            tok, tag = line.split("\t")
            sent.append((tok, tag))
    if sent: sents.append(sent)
    return sents
