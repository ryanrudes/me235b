import typer

from detector import detect
from drawing import run_draw
from grasp_from_tag import grasp_transform
from hanoi import run_hanoi
from urdf_visualizer import visualize_urdf

app = typer.Typer()
app.command("detect")(detect)
app.command("visualize")(visualize_urdf)
app.command("draw")(run_draw)
app.command("lab3")(run_hanoi)
app.command("grasp-transform")(grasp_transform)

if __name__ == "__main__":
    app()
