import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "src" / "app.py"), run_name="__main__")
