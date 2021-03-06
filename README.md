# Meta-Learning for Simple Regret Minimizatin

## Requirements:

`conda env create -f msrm.yml`

`source activate meta-srm`

## Usage
Run `python AdaTS_Public.py` and `python CoAdaTS_Public.py` for the Gaussian and Linear bandits experiments, respectively.

## Disclaimer
This code is based on the codes
from https://openreview.net/forum?id=Mj6MVmGyMDb and 
https://openreview.net/forum?id=5Re03X8Iigi

## Note
Please use parr = 1 for multiprocessing only if your machine has large enough memory 
as otherwise you run into plugging issues which return weird results. 

## Citation

If this work is helpful in your research, please consider citing:  

```bibtex
@article{azizi2022metalearning,
      title={Meta-Learning for Simple Regret Minimization}, 
      author={Mohammadjavad Azizi and Branislav Kveton and Mohammad Ghavamzadeh and Sumeet Katariya},
      year={2022},
      eprint={2202.12888},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Thank you
