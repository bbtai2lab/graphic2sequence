# graphic2sequence
---

pix2code implementation with image caption method from

[https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)

with 3 kinds of CNN encoder:
1. Resnet18
2. Resnet152 (`torch.no_grad()`)
3. 2 layer CNN pretrained by AutoEncoder (model is defined in AE_bootstrap.py)

---
AutoEncoding bootstrap images

Several experiment (*AE_experiment_v?.py*) to see **whether we can solve the button color issue** by pretraining a CNN-Encoder by Auto-Encoding bootstrap images.

conclusion : Failed.

