# Summary of results in 2018-09
---

All experiment Codes, Logs and saved Models can be found in folder `AWS:~/bbtai2/featuremap/`.

## featuremap

in order to seperate color space of "blue", "black", "white", "gray", "orange", "yellow", "red", we trained a 2 layer CNN model with [*find_color_weights.py*](./featuremap/find_color_weights.py). the result is shown as following,
![conv_color.png](./featuremap/figures/conv_color.png)

then we pretrained a CNN model by using *autoencoder* with [AE_precondition_release3.py](./featuremap/AE_precondition_release3.py). in this model, `conv_color` is used as the highest level layer (fixed during training) to make sure that CNN is able to identify color feature.

after that the pretrained `Encoder` model in [AE_precondition_release3.py](./featuremap/AE_precondition_release3.py) is used as the CNN part in `EncoderCNN` in [pix2code_v2_20.py](./featuremap/pix2code_v2_20.py), where their weights are fixed during training. the loss curve looks like,

![losscurve_pix2code_v2_20.png](./featuremap/figures/losscurve_pix2code_v2_20.png)
