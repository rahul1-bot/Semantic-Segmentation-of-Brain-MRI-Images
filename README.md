# _Semantic Segmentation of Brain MRI Images for FLAIR Abnormailty Detection_

## _Authors_
* Rahul Sawhney <sup>1</sup>
* Aabha Malik <sup>1</sup>
* Shubham Garg <sup>1</sup>
* Harsh Sharma <sup>1</sup>

## _Abstract_
Brain Tumor segmentation is an essential method in medical image processing. Early medical diagnosis of brain tumors plays an essential function in enhancing treatment possibilities and increases the survival rate of the patients. The most challenging and time taking work is the manual segmentation of the brain tumors for cancer diagnosis from large quantity of MRI images produced in scientific routine. Recently, automated segmentation utilizing deep learning techniques showed popular considering that these techniques accomplish cutting edge outcomes and resolve this issue much better than other approaches. Deep Learning techniques can likewise make it possible for effective processing and unbiased assessment of the big quantities of MRI-based image information. There are variety of existing evaluation papers, concentrating on conventional techniques for MRI-based brain tumor image segmentation. In this paper, we have used customized UNet, FPN and ResNext Unet model along with this the AdamW and AdaMax are used as optimizer. Criterion used is IOU (Intersection Over Union) and the metric utilized for evaluation is Dice Score. 

## _Keywords_
* Artificial Neural Networks
* Convolutional Neural Networks
* Semantic Segmentation
* Image and Pattern Recognition
* Brain Tumor segmentation 

## _Methodology_
![methodology](https://user-images.githubusercontent.com/65220704/173177275-fd6aaf6f-c5b1-44fb-9c38-c4e519e12adc.png)

## _Models Architecture_
* ### _Customized UNet_
![unet_2](https://user-images.githubusercontent.com/65220704/173177603-a49dc656-54d0-41a4-aa9c-8d0307300a3f.png)

* ### _Customized Feature Pyramid Network_
![fpn_2](https://user-images.githubusercontent.com/65220704/173177643-a150e79f-4f23-47e7-a3eb-daa61b3c0ee7.PNG)
 
* ### _Customized SE-ResNeXT Network_ 
![resnext](https://user-images.githubusercontent.com/65220704/173177519-93676910-2141-4cca-9696-1173a606599b.png)

## _Results_

| Models   | Mean Loss on Train | Mean Loss on Validation | Mean Dice on Train | Mean Dice on Validation | Mean IOU on Test  |
| -------- | ------------------ | ----------------------- | ------------------ | ----------------------- | ----------------- |
| ResNeXT  | 0.158              | 0.166                   | 0.859              | 0.938                   | 0.923             |
| FPN      | 0.255              | 0.301                   | 0.792              | 0.823                   | 0.812             |
| UNET     | 0.384              | 0.381                   | 0.753              | 0.763                   | 0.752             |
