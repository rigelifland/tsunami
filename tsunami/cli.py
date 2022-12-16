"""Console script for tsunami."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("tsunami")
    click.echo("=" * len("tsunami"))
    click.echo("A library for storing large wavs")


if __name__ == "__main__":
    main()  # pragma: no cover
