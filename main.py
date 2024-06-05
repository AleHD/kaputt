import re
from argparse import ArgumentParser

from kaputt.investigate import investigate


def parse_nodelist(nodelist: str) -> tuple[str, list[str]]:
    host, suspects = re.match(r"^(.*)\[(.*)\]$", nodelist).groups()
    return host, suspects.split(",")


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")

    investigate_parser = subparsers.add_parser("investigate")
    investigate_parser.add_argument("nodelist", type=parse_nodelist)

    args = parser.parse_args()
    if args.action == "investigate":
        host, suspects = args.nodelist
        investigate(host, suspects)
