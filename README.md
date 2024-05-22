# ATSim3D_pub

## Public executable files and test cases of ATSim3D [publised at ISEDA24]

### Intro: 

Thermal simulation plays a fundamental role in the thermal design of integrated circuits, especially 3D-ICs. Current simulators require significant runtime for high-resolution simulation, and dismiss the complex nonlinear thermal effects, such as nonlinear thermal conductivity and leakage power. 
    
To address these issues, we propose ATSim3D, a thermal simulator for simulating the steady-state temperature profile of nonlinear and heterogeneous 3D IC systems. This repo includes [all the test cases] and [part of the source codes] of ATSim3D, while the remaining codes are not available for commercial reasons (in collaboration with industry).

Detailed algorithms and details can be found in our paper, which can be cited as:

    @inproceedings{wang2024atsim, 
        title={ATSim3D: Towards Accurate Thermal Simulator for Heterogeneous 3D IC Systems Considering Nonlinear Leakage and Conductivity},
        author={Wang, Qipan and Zhu, Tianxiang and Lin, Yibo and Wang, Runsheng and Huang, Ru},
        booktitle={2024 International Symposium of Electronics Design Automation (ISEDA)},
        pages={1--6},
        year={2024}
    }

### Requirements:

0. Python == 3.6

1. Numpy >= 1.19

2. pandas >= 1.1

3. scipy >= 1.5

4. tqdm

5. Other common packages in python, including csv, os, math, matplotlib, etc. 


### Usage

General:

python ATSimCore.py

1. [2DIC]

python ATSim3D.py --lcfFile ../2DIC/Intel_ID1_lcf.csv --ConfigFile ../2DIC/Intel.config --SimParamsFile ../2DIC/SimParms.config

2. [Mono3D]

python ATSim3D.py --lcfFile ../Mono3D/Mono3D_lcf.csv --ConfigFile ../Mono3D/Mono3D.config --SimParamsFile ../Mono3D/SimParms.config

3. [TSV3D]

python ATSim3D.py --lcfFile ../TSV3D/TSV3D_lcf.csv --ConfigFile ../TSV3D/TSV3D.config --SimParamsFile ../TSV3D/SimParms.config

    
