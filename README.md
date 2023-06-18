# Lightning Template

```
copper_train --help
```

examples

- `copper_train data.num_workers=16`
- `copper_train data.num_workers=16 trainer.deterministic=True +trainer.fast_dev_run=True`

## Development

Install in dev mode

```
pip install -e .
```

### Docker

# For training
python src/train.py experiment=cifar10_example

docker build -t anmol9696/emlo4 -f dockers/train/Dockerfile .
# or pull from Docker hub
docker pull anmol9696/emlo4:latest

# For training through docker 
docker run -it --volume /workspace/lightning-hydra-timm/dockerMount:/opt/src/logs  anmol9696/emlo4 python src/train.py experiment=cifar10_example

# For evaluating through docker
docker run -it --volume /workspace/lightning-hydra-timm/dockerMount:/opt/src/logs  anmol9696/emlo4 python src/eval.py experiment=cifar10_example
