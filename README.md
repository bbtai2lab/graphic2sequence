# graphic2sequence
---

pix2code implementation with image caption method from

[https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)

with 3 kinds of CNN encoder:
1. Resnet18 [./pix2code_resnet18_free.py](./pix2code_resnet18_free.py)
2. Resnet152 (`torch.no_grad()`) [./pix2code_resnet152_fixed.py](./pix2code_resnet152_fixed.py)
3. 2 layer CNN pretrained by AutoEncoder [./pix2code_AE_pretrained.py](./pix2code_AE_pretrained.py)
> model is defined in [./AE_bootstrap.py](./AE_bootstrap.py)

**conclusion** : best accuracy is acheived by *free* Resnet18 CNN-Encoder ( *free* : train CNN-Encoder and LSTM-Decoder simultaneously.)

# other files

- [./memo.md](./memo.md) : Log of experiments
- [./Library_v1.py](./Library_v1.py) : Vocabulary and Dataset class
- bootstrap.vocab : vocabulary for bootstrap


# AutoEncoding bootstrap images
---

Several experiments in [./AE_experiment](./AE_experiment) to see **whether we can solve the button color issue** by pretraining a CNN-Encoder by Auto-Encoding bootstrap images.

conclusion : Failed.

