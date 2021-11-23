# MS-G3D

This repository, used to run the MS-G3D model as a baseline to the Conflab dataset paper is a fork of the original repository by the MS-G3D authors:

https://github.com/kenziyuliu/MS-G3D

We have added the conf files and data loaders necessary to fine-tune and test with the Conflab skeleton data.

Fine tune the model with:

```
python3 main.py --config ./config/conflab/train_joint.yaml --work-dir transfer_learning/conflab/joint --weights pretrained-models/kinetics-joint.pt --ignore-weights fc.weight fc.bias --half
```

And test with:

```
python3 main.py --config ./config/conflab/test_joint.yaml --work-dir transfer_learning/conflab/joint_test --weights transfer_learning/conflab/joint/weights-4-544.pt
```

Due to their size, trained weights are not in the repository.