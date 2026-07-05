import typer
import pickle
from pathlib import Path
from typing import Optional

# Ensure user has installed full extras before importing data/viz
try:
    import pandas as pd
    from systemictau.data import from_dataframe
    from systemictau.visualization import plot_tau_evolution
except ImportError:
    raise ImportError("CLI requires the 'full' installation. Run 'pip install systemictau[full]'")

app = typer.Typer(help="Systemic Tau Command Line Interface")

@app.command()
def analyze(
    data_path: Path = typer.Argument(..., help="Path to input CSV data file"),
    output: Path = typer.Option(..., "--output", "-o", help="Path to save the output SystemicTauResult (.pkl)"),
    time_col: Optional[str] = typer.Option(None, "--time-col", "-t", help="Column name to use as time/index"),
    window_size: int = typer.Option(13, "--window", "-w", help="Size of the sliding window"),
    imputation: str = typer.Option('linear', "--imputation", "-i", help="Missing data handling ('linear', 'ffill', 'drop')")
):
    """
    Computes Systemic Tau from a CSV dataset and saves the result to a pickle file.
    """
    if not data_path.exists():
        typer.secho(f"Error: File {data_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
        
    typer.secho(f"Loading data from {data_path}...", fg=typer.colors.CYAN)
    df = pd.read_csv(data_path)
    
    typer.secho(f"Computing Systemic Tau (window={window_size}, imputation={imputation})...", fg=typer.colors.CYAN)
    result = from_dataframe(df, time_col=time_col, window_size=window_size, imputation=imputation)
    
    # Save result using pickle
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'wb') as f:
        pickle.dump(result, f)
        
    typer.secho(f"Success! Result saved to {output}", fg=typer.colors.GREEN)

@app.command()
def plot(
    result_path: Path = typer.Argument(..., help="Path to the saved SystemicTauResult (.pkl)"),
    plot_type: str = typer.Option('tau_evolution', "--type", help="Type of plot: 'tau_evolution'")
):
    """
    Plots results from a previously computed SystemicTauResult.
    """
    if not result_path.exists():
        typer.secho(f"Error: File {result_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
        
    with open(result_path, 'rb') as f:
        result = pickle.load(f)
        
    import matplotlib.pyplot as plt
    
    if plot_type == 'tau_evolution':
        typer.secho("Generating Systemic Tau Evolution plot...", fg=typer.colors.CYAN)
        plot_tau_evolution(result.taus_global)
        plt.show()
    else:
        typer.secho(f"Error: Unknown plot type '{plot_type}'", fg=typer.colors.RED)
        raise typer.Exit(code=1)

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Host IP to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind")
):
    """
    Starts the Systemic Tau REST API (FastAPI backend).
    """
    try:
        import uvicorn
        from systemictau.platform.api.main import app as api_app
    except ImportError:
        typer.secho("FastAPI/Uvicorn not installed. Run 'pip install systemictau[platform]'", fg=typer.colors.RED)
        raise typer.Exit(code=1)
        
    typer.secho(f"Starting Systemic Tau API on http://{host}:{port}", fg=typer.colors.GREEN)
    uvicorn.run(api_app, host=host, port=port)

@app.command()
def ui():
    """
    Launches the interactive Systemic Tau Dashboard (Streamlit).
    """
    import subprocess
    import sys
    
    try:
        import streamlit  # noqa: F401
    except ImportError:
        typer.secho("Streamlit not installed. Run 'pip install systemictau[platform]'", fg=typer.colors.RED)
        raise typer.Exit(code=1)
        
    from systemictau.platform.dashboard import app as dash_app
    app_path = Path(dash_app.__file__)
    
    typer.secho("Starting Systemic Tau UI...", fg=typer.colors.GREEN)
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])

plugin_app = typer.Typer(help="Manage Systemic Tau plugins.")
app.add_typer(plugin_app, name="plugin")

@plugin_app.command("create")
def create_plugin(name: str):
    """
    Scaffolds a new Systemic Tau plugin with pluggy entrypoints.
    """
    import os
    plugin_dir = f"systemictau-plugin-{name}"
    os.makedirs(plugin_dir, exist_ok=True)
    
    # setup.py
    with open(f"{plugin_dir}/setup.py", "w") as f:
        f.write(f'''from setuptools import setup, find_packages

setup(
    name="systemictau-plugin-{name}",
    version="0.1.0",
    packages=find_packages(),
    entry_points={{
        "systemictau": [
            "{name} = {name}.hooks",
        ],
    }},
)
''')

    # Python module
    os.makedirs(f"{plugin_dir}/{name}", exist_ok=True)
    with open(f"{plugin_dir}/{name}/__init__.py", "w") as f:
        pass
        
    with open(f"{plugin_dir}/{name}/hooks.py", "w") as f:
        f.write('''import systemictau
import numpy as np

@systemictau.hookimpl
def compute_custom_correlation(X: np.ndarray) -> np.ndarray:
    """
    Replace Kendall Tau with a proprietary metric.
    """
    return np.corrcoef(X, rowvar=False)
''')

    typer.echo(f"Successfully created plugin scaffold at ./{plugin_dir}")
    typer.echo("To install it, run:")
    typer.echo(f"  cd {plugin_dir} && pip install -e .")

if __name__ == "__main__":
    app()
