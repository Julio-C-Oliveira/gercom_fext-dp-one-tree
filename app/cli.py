import argparse

from app.run import run_simulation

def main():
    parser = argparse.ArgumentParser(
        description="fext-dp: Federated Learning for Decision Trees with Differential Privacy"
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="Roda o experimento com as configurações definidas no pyproject.toml"
    )

    args = parser.parse_args()

    if args.run:
        run_simulation()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()