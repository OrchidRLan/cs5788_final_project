"""Training entry point."""
import argparse
import sys
sys.path.insert(0, ".")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()

    from triposg_finetune.trainer import train
    train(args.config)

if __name__ == "__main__":
    main()
