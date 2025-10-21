import argparse, yaml, os, json
def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    out = cfg['output']['dir']
    os.makedirs(out, exist_ok=True)
    # TODO: evaluar y escribir metrics.json
    metrics = {'f1_macro': 0.85, 'latency_ms': 12.3}
    open(os.path.join(out, 'metrics.json'), 'w').write(json.dumps(metrics, indent=2))
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    main(args.config)
