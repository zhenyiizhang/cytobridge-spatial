"""Compatibility entry point for the authoritative ARISTA calculation notebook."""
from build_arista_reader_tutorials import dataset


def build_notebook():
    return dataset()


if __name__ == "__main__":
    build_notebook()
