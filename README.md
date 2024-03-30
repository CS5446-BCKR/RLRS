# RLRS

## Installation

- python=3.10
- `pip install -r requirements.txt`

## Install Pytorch
- Follow instruction from the website.
- In my case: `conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch`.

Finally, install the package: `pip install -e .`

# Development

## Test
```bash
make test
```

## Formatting
```bash
make fmt
```

## Debugging Training
```bash
make test-train
```

## Loss Visualization

1. Start mlflow server:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

Open `127.0.0.1:8080` to view MLflow, choose `Experiments`, select the run, click on **Model Metrics** tab to see all losses, metrics.
