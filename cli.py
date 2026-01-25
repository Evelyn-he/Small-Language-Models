import argparse
from src.main import main_loop

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true"
    )

    parser.add_argument(
        "--generate_data",
        "-gd",
        action="store_true"
    )

    args = parser.parse_args()

    main_loop(args)

if __name__ == "__main__":
    main()