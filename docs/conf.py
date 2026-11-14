import subprocess
from pathlib import Path

project = "spatial-casadi"
author = "Christopher E. Mower"
release = "2.0.0"
extensions = ["sphinx.ext.autodoc", "breathe"]
html_theme = "furo"
html_title = project
html_logo = "../doc/image/spatial-casadi.svg"
html_favicon = "../doc/image/spatial-casadi.svg"
html_show_sphinx = False
html_static_path = ["_static"]
html_css_files = ["style.css"]
exclude_patterns = ["_build"]
autodoc_member_order = "bysource"
root = Path(__file__).resolve().parent
(root / "_build/doxygen").mkdir(parents=True, exist_ok=True)
subprocess.run(["doxygen", "Doxyfile"], cwd=root, check=True)
breathe_projects = {"spatial-casadi": str(root / "_build/doxygen/xml")}
breathe_default_project = "spatial-casadi"
