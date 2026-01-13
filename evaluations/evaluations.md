# Evaluations

This folder contains the Federated Learning evaluation scripts and to run the experiments described in the paper.

We use the Flower framework for Federated Learning to implement the evaluations. More details on Flower can be found at https://flower.dev/.

## Run the evaluations

To run an evaluation,

```bash
cd evaluations
uv venv
source .venv/bin/activate # use activate.fish for fish shell
```

It is important to install the correct dependencies because we rely for these experiments on specific versions of Flower. 