# FastRec

This repo serves as the appendix artifact of the final project of CSCI 5570.

To run our code, you can first clone our repository to your local directory:
```shell
$ cd \path\to\your\directory
$ git clone https://github.com/WeiranHuang/FastRec.git
```
Anaconda is recommended to monitor your packages and environments. You can download it [here](https://www.anaconda.com/) with full instructions. After a successful installation (you can check this with command `conda --version`),
you can create a virtual environment for this project. We also prepared a `requirements.txt` to list all our dependencies, thus you can utilize it to create an identical environment with the following command:

```shell
$ cd \path\to\FastRec
$ conda create --name <env> --file requirements.txt
```

If the dependencies can be installed successfully, you should be able to run our code. You can use:
```shell
$ python main.py --help
```
to see all arguments. But because the argument parser is copied from the DLRM codebase, some arguments can be redundant.

A sample run is:
```shell
$ python main.py --queue --num_batches 4 --inference_engines 4
```
