# Deep Learning

Environment for the TensorFlow exercises and image-classification notebooks.
The model and data modules are placeholders for later work.

## Setup

Use Python 3.10 through 3.13. From this directory, create an environment and
install the project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python test_environment.py
python -m tox -e py311
python -m jupyter lab
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.
The notebooks currently contain no cells; their content will be added later.

TensorFlow uses the CPU on native Windows. GPU support for current TensorFlow
releases requires WSL2 or a supported Linux setup.
