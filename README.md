


# STK-former: Efficient Image Inpainting Model Based on TopK Sparse Attention and Dynamic K-value Regulation




</table>

Example completion results of our method on images of face ([CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)), building ([Paris](https://github.com/pathak22/context-encoder)), and natural scenes ([Places2](http://places2.csail.mit.edu/)) with center masks (masks shown in gray). For each group, the masked input image is shown left, followed by sampled results from our model without any post-processing. The results are diverse and plusible.



# Installation
This code was tested with Pytoch 1.10.0, CUDA 11.3, Python 3.6 


# Datasets
- ```face dataset```: 24183 training images and  2824 test images from [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and use the algorithm of [Growing GANs](https://github.com/tkarras/progressive_growing_of_gans) to get the high-resolution CelebA-HQ dataset
- ```building dataset```: 14900 training images and 100 test images from [Paris](https://github.com/pathak22/context-encoder)
- ```natural scenery```: original training and val images from [Places2](http://places2.csail.mit.edu/)
- ```object``` original training images from [ImageNet](http://www.image-net.org/).


# License
<br />This work is licensed under a <a rel="license" href="https://github.com/lyndonzheng/Pluralistic-Inpainting"> License</a>.https://github.com/lyndonzheng/Pluralistic-Inpainting


# Citation

If you use this code for your research, please cite our paper.
```

@article{1123,
  title={STK-former: Efficient Image Inpainting Model Based on TopK Sparse Attention and Dynamic K-value Regulation},
  author={},
  journal={},
  pages={},
  year={},
  publisher={}
}
```
