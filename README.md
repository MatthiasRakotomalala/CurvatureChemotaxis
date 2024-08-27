# Curvature in Chemotaxis: A Model for Ant Trail Pattern Formation

This Git repository contains the Python source code for the numerical schemes used in the paper: https://arxiv.org/abs/2408.13363, where the authors introduced a continuous ant trail pattern formation model. Please refer to the linked preprint for a detailed presentation of the model and the numerical schemes.

The repository includes a Jupyter notebook implementing the Finite Difference scheme for a simplified system. Additionally, a Python app is provided to run the Monte Carlo particle scheme.

### Running the Monte Carlo Scheme

You can use the following command to run the Monte Carlo scheme with specific parameters:

```
python AntSim_parser.py --N 1000 --Nt 500 --T 2 --NFr 35 --tau .1 --createVideo
```
For a detailed description of all available parameters, use the `--help` option:
```
python AntSim_parser.py --help
```

### Examples of Monte Carlo numerical results
![MonteCarloParticleFourierSimMltpTimes014740pp](https://github.com/user-attachments/assets/035b0b4f-c61e-4b31-a5cb-a975269da655)
![MonteCarloParticleFourierSimMltpTimes](https://github.com/user-attachments/assets/2cc49388-40c4-45e1-bea8-ca4fcace08c8)
