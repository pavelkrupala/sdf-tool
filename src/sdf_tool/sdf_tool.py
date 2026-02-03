import sys
import argparse

def main():
    parser = argparse.ArgumentParser(prog="sdf-tool")
    args = parser.parse_args()

    print("Hello, World!")

    return 0


if __name__ == '__main__':
    sys.exit(main())