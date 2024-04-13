# RLRS

## Installation

- python=3.10
- `pip install -r requirements.txt`

## Install Pytorch
- Follow instruction from the website.
- In my case: `conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch`.

Finally, install the package: `pip install -e .`

# Development

## Unit Test
```bash
make test
```

## Formatting
```bash
make fmt
```


## Loss Visualization

** To run any training, we need to start `mlflow` first. **

1. Start mlflow server:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

Open `127.0.0.1:8080` to view MLflow, choose `Experiments`, select the run, click on **Model Metrics** tab to see all losses, metrics.


## Debugging Training

### MovieLen training
```bash
make mock-train-movie
```

### Ayampp training
```bash
make mock-train-food
```

# Model Inference

## Start the server
```bash
make server
```

Go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to check how to call API(s).

# Evaluation

```bash
python scripts/eval.py <config-file> [--verbose]
```
