# TODO:
- mini-batch?
- inne funkcje aktywacji
- inne funkcje straty
- testy/eksperymenty
- podział zbioru na train/val/test
- lepsza inicjalizacja wag
- konfigurowalne biasy

# Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install in editable mode
pip install -e .

# 3. (Optional) Install Jupyter in this venv
pip install ipykernel jupyterlab

# 3. Install ipykernel
pip install ipykernel

# 5. Register kernel for Jupyter
python -m ipykernel install --user --name=venv --display-name "Projekt 3 - Weather prediction"

# 6. Start Jupyter Lab
jupyter-lab

# 7. Verify installation
# Open the notebook: notebooks/test_utils.ipynb
# Select the "Python (MLP)" kernel and run.
# If you see message printed, your setup is correct.
```
