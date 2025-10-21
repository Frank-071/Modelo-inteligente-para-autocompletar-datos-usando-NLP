import argparse, os, json, matplotlib.pyplot as plt
def main(results_dir):
    os.makedirs('docs', exist_ok=True)
    # Demo: leer metrics.json si existe y graficar una barra simple
    path = os.path.join(results_dir, 'metrics.json')
    if os.path.exists(path):
        metrics = json.load(open(path))
        plt.figure()
        plt.bar(list(metrics.keys()), list(metrics.values()))
        plt.title('Resultados (demo)')
        plt.savefig('docs/demo_plot.png', bbox_inches='tight')
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    args = ap.parse_args()
    main(args.results_dir)
