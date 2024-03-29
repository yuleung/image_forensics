## image_forensics
This repository is the code of the paper:  "*No one can escape: A general approach to detect tampered and generated image*"
## Environment
tensorflow 1.13.0, python3.6.8, cuda 10.0, cudnn 7.4

Other packages: cv2，skimage，numpy

Please run the following command to install those modules:
```
    pip install opencv-python
    pip install skimage
    pip install numpy
```
## Datasets

`CACIA2.0`: CACIA2.0 dataset contains 7491 authentic and 5123 tampered color images. The images in this database are with different size, various from 240×160 to 900 × 600 pixels, and the images have different formats: BMP,TIFF and JPEG images which with different Q factors.In this database, images have different scenes, and the tampered area has different post-processing. Speciﬁcally, some operations include resizing, deforming, and rotating taken to the splicing region before pasting to a ﬁnal generation. Moreover, blurring operation are done along the boundary area of the tampered region or other than the boundary area of the tampered region. This database can represent most of the splicing tampering operations in real life. 

`GPIR dataset`: GPIR dataset contains 80 images with realistic copy-move forgeries. All these images have size 768×1024 pixels, while the forgeries have arbitrary shapes aimed at obtaining visually satisfactory results.

`COVERAGE dataset`: COVERAGE contains 100 original-forged image pairs where each original contains multiple similar-but-genuine objects. It makes the discrimination of forged from genuine objects highly challenging. And six types of tampering were employed for forged image generation. 

`BigGANs dataset`: We used the BigGANs generator pre-trained on the ImageNet Dataset with truncation threshold 0.4 to generated 1000 categories of images (16 images for each category, including 8 images with the resolution of 256×256 and 8 images with the resolution of 512×521, for a total of 16,000 images). The pre-trained mode downloaded from TensorFlow-Hub (https://tensorflow.google.cn/hub/). 

`LSUN Bedroom dataset (256×256)`: LSUN dataset contains around one million labeled images for each of 10 scene categories and 20 object categories. In our experiments, we selected images with the resolution of 256×256 in the scene categories of the bedroom from this dataset. 

`PGGAN dataset`: We downloaded 10,000 images of the bedroom generated by PGGANs which trained on LSUN dataset, and the resolution of the images is 256×256. In addition, we downloaded 169 selected high-quality images generated by PGGAN, of which 71 have the resolution of 256×256 and the remaining 98 have the resolution of 1024×1024. These images were released by NVIDIA at https://github.com/tkarras/progressive_growing_of_gans/ 

`SNGAN dataset`: SNGAN was proposed in 2018. We downloaded 7150 images of the dog and cat which generated by SNGANs, and images with the resolution of 128×128 and images with the resolution of 256×256 are half each. These images were released at https://github.com/pfnet-research/sngan_projection/ 

`StyleGAN dataset`: StyleGAN was proposed in 2019. It works well when generate a single object. we downloaded 10,000 images of the bedroom generated by StyleGANs which trained on LSUN dataset, and the resolution of the images is 256×256. These images were released by NVIDIA at https://github.com/NVlabs/stylegan/

## Prepare data for training and test
First we need to divide the dataset into training and test sets：

`Case 1`:
* trianing set：CASIA 2.0 + BigGANs       
* test set： CASIA2.0、images generated by various GANs models（BigGANs、PGGAN、SNGAN、StyleGAN）

Randomly selected 5123 images from 7491 real images and 5123 tampered images from CASIA 2.0, and 5123 images were randomly selected from the 16,000 images of BigGANs dataset. Then, randomly selected 4123 images from each class, and total 12,369 images as the `training set`, and the remaining 1000 images from each class and total of 3000 images as the `test set(CASIA2.0、BigGANs)`. To test generalization for GANs, use the total images generated by PGGAN、SNGAN、StyleGAN as the `test set`

`Case 2`:
* trianing set：CASIA 2.0      
* test set： CASIA2.0

As Case 1, just the training set and test set only contain CASIA2.0

`Case 3`
* trianing set：CASIA 2.0 + BigGANs(pretrain)  COVERAGE(fine-tune) GPIR(fine-tune)      
* test set： COVERAGE、GPIR

we ﬁrst evaluate our approach on two datasets separately and then perform cross evaluation on the two datasets(one dataset as training and other as testing).

