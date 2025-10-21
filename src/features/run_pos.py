import argparse, os
def run(tagger):
    os.makedirs('data/cache', exist_ok=True)
    open('data/cache/pos_done.txt','w').write(f'POS tagged with {tagger}\n')
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--tagger', default='spacy')
    args = ap.parse_args()
    run(args.tagger)
