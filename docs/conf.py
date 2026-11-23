# -- Path setup --------------------------------------------------------------
import importlib
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "neuromodes"
copyright = "2025, Monash University"
author = "Neural Systems and Behaviour Lab"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "myst_nb",
    "sphinx.ext.mathjax",  # For rendering equations
    "sphinx.ext.linkcode",  # For linking to GitHub source code
    "sphinx.ext.napoleon",  # For better formatting of docstrings
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "tags": "ams",
        "useLabelIds": True,
    }
}

nb_execution_mode = 'off'

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True
templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_show_sourcelink = False
html_logo = '_static/logo.png'
html_theme_options = {'logo_only': True}
html_static_path = ['_static']
html_css_files = ['custom.css']

def linkcode_resolve(domain, info):
    """Function to link code to GitHub source files."""
    # Only build source links for Python API objects.
    if domain != "py":
        return None

    module_name = info.get("module")
    full_name = info.get("fullname")
    if not module_name or not full_name:
        return None

    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001
        return None

    # Walk from the imported module to the exact object Sphinx is documenting,
    # such as a class method like ``EigenSolver.compute_lbo``.
    obj = module
    for part in full_name.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None

    try:
        # Get both the file path and the line range so GitHub can open the
        # source at the correct object definition.
        file_path = inspect.getsourcefile(obj)
        source_lines, start_line = inspect.getsourcelines(obj)
    except Exception:  # noqa: BLE001
        return None

    if not file_path:
        return None

    # Convert the absolute file location into a repository-relative path for
    # the GitHub blob URL.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    rel_path = os.path.relpath(file_path, repo_root).replace(os.path.sep, "/")
    end_line = start_line + len(source_lines) - 1

    # Prefer exact commit on RTD, fallback to branch
    git_ref = (
        os.environ.get("READTHEDOCS_GIT_COMMIT_HASH")
        or os.environ.get("READTHEDOCS_VERSION")
        or "main"
    )
    if git_ref == "latest":
        git_ref = "main"

    return f"https://github.com/NSBLab/neuromodes/blob/{git_ref}/{rel_path}#L{start_line}-L{end_line}"
