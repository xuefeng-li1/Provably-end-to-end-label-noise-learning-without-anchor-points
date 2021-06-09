#VolMinNet

##Provably End-to-end Label-noise Learning without Anchor Points 

Xuefeng Li, Tongliang Liu, Bo Han, Gang Niu, Masashi Sugiyama.


##PyTorch implementation

###Dependencies

we implement our methods by PyTorch on NVIDIA Tesla V100. The environment is as bellow:

- [PyTorch](https://PyTorch.org/), version >= 1.7.1
- [CUDA](https://developer.nvidia.com/cuda-downloads), version >= 11.1


###Install PyTorch and Torchvision (Pip3):

pip3 install torch torchvision

###Experiments

We verify the effectiveness of VolMinNet on three synthetic noisy datasets (MNIST, CIFAR-10, CIFAR-100), and one real-world noisy dataset (clothing1M). And We provide [datasets](https://drive.google.com/drive/folders/1OYsRH9x37LQhbmGNv-1Ao1iYTHQN8W7F?usp=sharing) (the images and labels have been processed to .npy format).


###To run the code:


python3 main.py --dataset &lt;-dataset-&gt;  --noise_type  &lt;-noise type-&gt; --noise_rate &lt;-noise rate-&gt; --save_dir &lt;-path of the directory-&gt;


Here is an example: 

python3 main.py --dataset mnist  --noise_type  flip --noise_rate 0.45 --save_dir tmp


The statistics will be saved in the specified directory (tmp in this example). 