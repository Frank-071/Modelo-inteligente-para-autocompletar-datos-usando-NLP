import argparse, yaml, os
def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    os.makedirs(cfg['output']['dir'], exist_ok=True)
    # TODO: cargar datos, features, entrenar modelo, guardar resultados
    open(os.path.join(cfg['output']['dir'], 'train_log.txt'), 'w').write('training placeholder\n')
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    main(args.config)
