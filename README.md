## Identifying 2D brain slices in a 2D reference atlas using Siamese Networks

This repository contains code for the following paper [[arXiv]](https://arxiv.org/abs/2109.06662). If you found it useful in your research, please consider citing us:

```
@misc{antanavicius2021identifying,
      title={Identifying partial mouse brain microscopy images from Allen reference atlas using a contrastively learned semantic space}, 
      author={Justinas Antanavicius and Roberto Leiras and Raghavendra Selvan},
      year={2021},
      eprint={2109.06662},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Dataset Directory Structure
        
	|-- data
	    |-- atlas            # The name of dataset
	        |-- 30.jpg       # The position of atlas plate
            |-- ...
        |-- train
            |-- 30.jpg       # The position of brain slice
            |-- ...
        |-- val
        |-- test

## Training

`% python trainers/train.py --help `
```
usage: train.py [-h] image_size

Train Siamese Networks with triplet semi-hard loss

positional arguments:
  image_size  The size of images (256 or 1024)

optional arguments:
  -h, --help  show this help message and exit
```

By default, the Siamese Networks use ResNet50v2 as a base network. Paths to the images are specified in `paths.py`



## Testing

`% python evaluate.py --help `
```
usage: evaluate.py [-h] [-v] image_size weigths

Evaluate trained ResNet50v2 on the test dataset

positional arguments:
  image_size       The size of images (256 or 1024)
  weights          Path to the model weights

optional arguments:
  -h, --help       show this help message and exit
  -v, --visualize  Visualize predicted atlas plates
```
## Limitations

 It is assumed that no more than one training example for each atlas plate is available. 
To include more training examples, `data_loader.py` has to be slightly modified.