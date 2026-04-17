import typer

from .detector import detect
from .urdf_visualizer import visualize_urdf
from .lab3 import run as run_lab3

app = typer.Typer()
app.command("detect")(detect)
app.command("visualize")(visualize_urdf)
app.command("lab3")(run_lab3)

if __name__ == "__main__":
    app()
