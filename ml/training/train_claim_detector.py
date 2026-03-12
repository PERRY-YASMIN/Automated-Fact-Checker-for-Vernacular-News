from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train claim detector model (Milestone 2 scaffold).")
    parser.add_argument("--data-path", type=str, default="", help="Path to labeled claim detector dataset")
    _ = parser.parse_args()
    print("Training scaffold ready: implement dataset loading and training loop here.")


if __name__ == "__main__":
    main()
