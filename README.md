
<p align="center">
    <i>The official Python client for the Neo Hub.</i>
</p>


## Welcome to the neohub library

The `neohub` library allows you to interact with the Neo Hub that contains hundreds of pre-trained models and datasets for your projects or play with the thousands of machine learning apps hosted on the Hub. 


## Installation

Install the `neohub` package with [pip](https://pypi.org/project/huggingface-hub/) using Nexus:

```bash
pip install neohub
```

## Quick start

### Download dataset

Download a dataset and load it

```py
from neohub import pull

dataset = pull(repo_id="aramco/AramcoPub",load=True)
```
