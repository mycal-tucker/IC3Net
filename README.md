# IC3Net

This repository contains reference implementation for IC3Net paper (accepted to ICLR 2019), **Learning when to communicate at scale in multiagent cooperative and competitive tasks**, available at [https://arxiv.org/abs/1812.09755](https://arxiv.org/abs/1812.09755)

## Cite

If you use this code or IC3Net in your work, please cite the following:

```
@article{singh2018learning,
  title={Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks},
  author={Singh, Amanpreet and Jain, Tushar and Sukhbaatar, Sainbayar},
  journal={arXiv preprint arXiv:1812.09755},
  year={2018}
}
```

## Standalone environment version

- Find `gym-starcraft` at this repository: [apsdehal/gym-starcraft](https://github.com/apsdehal/gym-starcraft)
- Find `ic3net-envs` at this repository: [apsdehal/ic3net-envs](https://github.com/apsdehal/ic3net-envs)

## Installation

First, clone the repo and install ic3net-envs which contains implementation for Predator-Prey and Traffic-Junction

```
git clone https://github.com/IC3Net/IC3Net
cd IC3Net/ic3net-envs
python setup.py develop
pip install tensorboardX
```


**Optional**: If you want to run experiments on StarCraft, install the `gym-starcraft` package included in this package. Follow the instructions provided in README inside that packages.


Next, we need to install dependencies for IC3Net including PyTorch. For doing that run:

```
pip install -r requirements.txt
```

## Running

Some example scripts have been provided for this. The code has been changed a bit from the original repo of IC3 Net
to support 2 things (1). Agents can now use discrete learnable prototypes for communication (2). Agents can now
train using a gating penalty(if specified) which enables agents to learn sparse communication protocols even
in fully cooperative scenarios. 

Also, the repo now uses tensorboard instead of visdom which can be used to view training plots. 

For discrete communication try out the script:

    python run_pp_proto.py

I would recommend going through the script once to better understand the arguments. This script learns discrete 
communication protocol with fixed gating head(g=1). After training, it also plots the graphs for rewards, success rates
and communication rates. 

Similarly, for trying out the gating penalty approach use 

    python run_g0.01.py


Similarity you can write training scripts for other environments. I am also including one for the traffic-junction 
environment. 

## Contributors

- Amanpreet Singh ([@apsdehal](https://github.com/apsdehal))
- Tushar Jain ([@tshrjn](https://github.com/tshrjn))
- Sainbayar Sukhbaatar ([@tesatory](https://github.com/tesatory))

## License

Code is available under MIT license.
