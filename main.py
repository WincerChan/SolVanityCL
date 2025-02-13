import multiprocessing

from core.cli import cli

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    cli()
