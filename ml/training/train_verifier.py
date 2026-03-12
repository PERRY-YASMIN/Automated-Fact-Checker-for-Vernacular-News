from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train verifier model (Milestone 2 scaffold).")
    parser.add_argument("--data-path", type=str, default="", help="Path to stance/NLI training dataset")
    _ = parser.parse_args()
    print("Training scaffold ready: implement verifier fine-tuning here.")


if __name__ == "__main__":
    main()
