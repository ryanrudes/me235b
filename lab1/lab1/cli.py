import typer

from .detector import detect


app = typer.Typer()
app.command()(detect)


if __name__ == "__main__":
    app()
