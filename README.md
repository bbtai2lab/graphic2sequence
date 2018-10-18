# graphic2sequence
---

pix2code implementation following image captioning models:

[https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)

## Experiments

### 201808

we train EncoderCNN+DecoderRNN with different CNN architectures:
1. Resnet18
2. Resnet152
3. 2 layer CNN pretrained by AutoEncoder

best validation loss is achieved by *free* Resnet18 architecture with `loss_valid -> 0.02` where overfitting becomes significant.

With `loss_valid -> 0.02`, we have many many mismatch in `btn-color`, and we don't know what is going wrong. Both "Color mismatch" and "Overfitting" become clear at *experiment_201809*

### 201809

Since layout of bootstrap image is very naive that we can even distinguish each layout only by the color of buttons, we pretrained a 2 layer CNN `conv_color` to seperate color channels, then each color feature will be put into 2~3 layer of CNN before embedding. Also, we encode code in *.gui* ourselves to some arrays first, and then train a DecoderRNN to see whether this Decoder can generate words perfectly.

In this series of experiment, We found that,

1. meadian batch size have better result.
> for example, `batch_size=16` is better than `batch_size=64`

2. SGD works better than Adaptive optimizer such as Adam and Adadelta.

3. simple CNN architecture is enough for bootstrap.
> even with `conv_color` only is enough.

4. mismatch of button color can be improved by a better EncoderCNN.

5. Overfitting mainly occurs in DecoderRNN.

In conclusion, this image captioning model is bad for realworld application, since it can only generate captions for layouts which have been already "seen" during training process. For a much more robust bootstrap captioning, one needs **UI detection**.
