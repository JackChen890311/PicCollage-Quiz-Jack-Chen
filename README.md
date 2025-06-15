# PicCollage Quiz
Author: Jack Chen  

This repository is for quiz for PicCollage.  
Problem Description is [here](https://docs.google.com/document/d/1SDjUvafOoxBjxCCtMns8GibLN-h_0nsgRzN93s1IbJw/edit?tab=t.0) (limited access).  

## Structure of this repository
- `dataloader.py`: Dataloader class specific designed for this problem  
- `model.py`: Model class, currently supporting Vanilla VAE and GAN  
- `trainer.py`: Trainer class for different kind of models  
- `main.py`: Main logic of end-to-end training and sampling  
- `config.yaml`: Configuration file for various parameters  
- `visualize.ipynb`: For quick visualization of the sampling results  
- `data/`: Input data  
- `output/`: Output data  
- `models/`: Model checkpoints  

## Environment
- Macbook M4 Pro
- Python 3.12
- Please refer to `requirements.txt` for packages

## Running the code
After configure the settings in `config.yaml`, for training and sampling, please run:
```bash
python main.py
```

For visualization, please use `visualize.ipynb`.