This repository provides the implementation of **DARE**, a data-aware Monte Carlo–based framework for estimating and explaining precision errors. It enables accurate error prediction, interpretable attribution of error sources, and efficient evaluation across diverse tensor operations.

---

## Environment Setup

We recommend using Python ≥ 3.9 and PyTorch ≥ 2.0 with CUDA support. The following dependencies are required:

```bash
# Create a fresh conda environment (optional)
conda create -n dare python=3.9 -y
conda activate dare

# Install PyTorch (adjust CUDA version if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies
pip install numpy matplotlib tqdm
```

Optional (for visualization and evaluation reports):

```bash
pip install seaborn pandas
```

---

## Directory Structure

The repository is organized as follows:

```
precision_estimation/
├── core/                         # Core implementation of DARE
│   ├── analyzer/                 # Error attribution and interpretability analysis
│   ├── config/                   # Precision strategy configurations
│   ├── detector/                 # Error detection pipeline
│   ├── generator/                # Input/weight/tensor data generators
│   └── oracle/                   # Monte Carlo oracle for error estimation
│
├── examples/                     # Examples for different operators
│   ├── conv2d_precision_pipeline.py
│   ├── layernorm_precision_pipeline.py
│   ├── matmul_precision_pipeline.py
│   └── ... (others: relu, softmax, pooling, attention, etc.)
│
├── evaluation/                   # Experimental evaluation framework
│   ├── experiment/               # Scripts for automated experiments
│   └── precision_evaluation_analyzer.py
│
├── *.log                         # Logs of precision experiments
├── *.png / *.svg                 # Visualization plots from evaluation
```

---

## Usage

### Running an Operator Pipeline

Each operator has a corresponding example script under `examples/`. For instance, to evaluate **Conv2D** with mixed-precision strategies:

```bash
python examples/conv2d_precision_pipeline.py
```

Similarly, you can run:

```bash
python examples/layernorm_precision_pipeline.py
python examples/matmul_precision_pipeline.py
python examples/softmax_precision_pipeline.py
```

### Running Evaluation Suite

The evaluation framework under `evaluation/` provides scripts to reproduce results across operators and strategies. For example:

```bash
python evaluation/precision_evaluation_analyzer.py
```

This generates aggregated metrics and plots under the project root, including:

* `plot_error_prediction_accuracy.*`
* `plot_bound_tightness_distribution.*`
* `plot_operator_efficiency.*`
* `plot_primary_error_sources.*`

### Output

* **Logs**: numerical results for each operator (`*_precision.log`)
* **Plots**: comparative figures in `.png` and `.svg` formats
* **Reports**: JSON/PNG summaries (`dare_evaluation_report.json`, `dare_evaluation_dashboard.png`)

