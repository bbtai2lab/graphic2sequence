# 実験の流れと考え
---

1. とりあえず、[https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning)
を実装して、100epochsを訓練してみる。
> folder : 2018-08-04-03-40-07; Resnet18, free, Adadelta

2. 1をもとに、さらに200epochsを学習させる。
> folder : 20180805-123547; Resnet18, free, Adam。これをもとに、さらに100epochs学習させる：folder : 20180806-125422; Resnet18, free, Adam
>
> folder : 20180805-123549; Resnet18, free, Adadelta。これをもとに、さらに100epochs学習させる：folder : 20180806-125458; Resnet18, free, Adadelta

結論
  - Adam：LOSS下がるのが早い；Adadelta：下がるのが遅いが、安定
  - optimizerが転移学習の時初期化された。optimizerの状態も引き継いだほうがいい？

3. Resnet152は果たして最強なのかを確認するために、Resnet18の代わりにResnet152を入れてみます。
> folder : 20180805-130004; Resnet152, fixed, Adam
> 
> folder : 20180805-130221; Resnet152, fixed, Adadelta

結論：
  -  Adam：LOSS下がるのが早いが、過学習しやすい？；Adadelta：下がるのが遅いが、安定
  -  Resnet152が複雑な画像識別に強いが、Fixして再訓練しないと、やはり再訓練したResnet18に敵わない　ー＞　HTML画像に最適したEncoderCNNが必要？
  
今までResnet18(free)の方が一番安定かつLoss低いように見えます。HTML画像に最適したEncoderCNNを探すために、Auto-Encoderのbootstrap画像の自分から自分を生成する訓練を使います.出発点になるモデルは
  ```
  class AE(torch.nn.Module):

    def __init__(self, num_latent, in_channels, num_feature):
        super(AE, self).__init__()

        self.encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, num_feature, kernel_size=4, stride=2, padding=1),
                    torch.nn.BatchNorm2d(num_feature),
                    torch.nn.LeakyReLU(0.2, inplace=True),

                    torch.nn.Conv2d(num_feature, num_feature, kernel_size=4, stride=2, padding=1),
                    torch.nn.BatchNorm2d(num_feature),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    )
        self.latent = torch.nn.Sequential(
                    torch.nn.Linear(num_feature*56*56, num_latent),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.Linear(num_latent, num_feature*56*56),
                    )

        self.decoder = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(num_feature, num_feature, kernel_size=4, stride=2, padding=1),
                    torch.nn.ReLU(inplace=True),

                    torch.nn.ConvTranspose2d(num_feature, in_channels, kernel_size=4, stride=2, padding=1),
                    torch.nn.Sigmoid(),
                    )
  ```
このモデルを100epochs訓練して、pix2codeに入れてみると、こんな感じ
  ```
  class EncoderCNN(torch.nn.Module):

    def __init__(self, embed_size):
        super(EncoderCNN,self).__init__()

        model_ = torch.load("./Models/AE_epoch_100.pkl")
        self.CNN = model_.encoder
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(in_features=model_.latent[0].in_features, out_features=embed_size),
            torch.nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, images):
        """Extract feature vectors from input images."""

        with torch.no_grad():
            features = self.CNN(images)

        features = features.reshape(features.size(0),-1)
        features = self.feature(features)
        return features
  ```
4. 以上は`latent`層を固定せずに再学習させるモデルです。結局すぐ過学習しました。`latent`中の`Linear`層の素子が多すぎるのが原因でしょうか？
> folder : 20180807-212125; AE, fixed, Adam

5. ということで、`latent`中の`Linear`層を全部再学習させないようにやってみます。結果として、よくなりました。
> folder : 20180808-182726; AE, fixed, Adam
```
class EncoderCNN(torch.nn.Module):

    def __init__(self, embed_size):
        super(EncoderCNN,self).__init__()

        model_ = torch.load("./Models/AE_epoch_100.pkl")
        self.CNN = model_.encoder
        modules = list(model_.latent.children())[:-1]
        self.latent = torch.nn.Sequential(*modules)
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(in_features=model_.latent[-1].in_features, out_features=embed_size),
            torch.nn.LeakyReLU(0.2, inplace=True),
            )
```

結論：
  - 訓練する中間Linear層のニューラル数がたくさんあると、過学習しやすい？
  - Train/Lossが低い、Test/Lossがなかなか落ちて来ないのはなぜでしょう？（過学習じゃない場合）
  
というわけで、AEモデルの精度によって、pix2codeに実装して使うときかなり影響受けるようです。もうちょっと調べます。

6. とりあえず、比較できるようにResnet18(free)を400epochs学習させる
> folder : 20180813-135108; Resnet18, free; Adadelta

7. AEを使う方のpix2codeの`self.feature`の中の`torch.nn.LeakyReLU(0.2, inplace=True),`は果たしているのか？単純に`torch.nn.BatchNorm1d(embed_size,momentum=0.01)`にしてみました。
> folder : 20180809-130653; AE, fixed; Adadelta

結論：
  - AutoEncoderを使った転移学習、LeakyReLUをBatchNorm1dに変えたらよくなったみたい。なぜ？もっといけそう？epochsを800するとLossがもっと下がるかな？
  - 両者のLoss(Train/Test)が同じ下がり方のように見える、Resnet18の方が速いかな。Resnet18は多分限界まで学習した？
  
8. 7を800epochs訓練すると、過学習。
> folder : 20180819-053738; AE, dixed; Adadelta

そして、ミスに気づきました。Auto-Encoderの学習に`torchvision.transforms.Normalize`がなかったが、pix2code+AEに`torchvision.transforms.Normalize`がありました。以下のように、

```
transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((crop_size, crop_size)), # Match resnet size
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

9. `torchvision.transforms.Normalize`を除いて800epochs訓練してみると、過学習が少ししか残っていない（最後に）。Test LossはやはりResnet18(free)より高い。
> folder : 20180819-234650; AE, fixed; Adadelta

疑問： 両方のTest/Lossの差は一体何が原因でしょうか？CNNの部分も一緒に訓練すること？

10. 9をもとに、`CNN`と`latent`--> free, embed_size --> 64, latent_size --> 64
> folder : 20180820-075257; AE, free; Adadelta

11. 9をもとに、`CNN`と`latent`--> free, embed_size --> 256, latent_size --> 256
> folder : 20180820-155838; AE, free; Adadelta

結論：
  - embed_sizeは「画像がこれからの時系列を語る」の多様性に対応するので、64はさすがに小さい
  - hidden_sizeはある一つの特徴を入力する時に出力の次元である、ある意味で、「次の入力特徴がそれの時系列を語る」の多様性に対応するので、hiddenz_size >= embed_size ?
  
というわけで、`LSTM`の方は多分これでままいけるでしょう。細かいところは多分`bidirection`とかつかって精度をあげるしかないかもしれない。button色の間違いはやはり`CNN`のfeature抽出を改善すれば治るでしょう。ここから`AE_experiment_v?.py`に入ります。

12. version 2, 画像生成に情報圧縮の`latent`層は必須ではないかもしれない、除いてみる。（pix2codeに実装しようと思うと、結局featureを代表する`Linear`層にまとめる形、つまり`latent`層になりますが、とりあえず、情報を捨てない場合色認識間違うかを確認する価値があります。）
> folder : 20180822-1358

結果：情報量が多いため、Loss下がるのが速いが、不安定。

13. version 3, 安定化するために、num_feature: 60 --> 30
> folder : 20180822-1502

結果：ほぼ同じレベルのLoss、num_featureが少なくなったせいなのか、黒字が少しぼやけて見える。

14. version 4, 真ん中にCNNx1を追加、さらに`BatchNorm2d`を追加。
> folder : 20180823-0343

結果：ほぼ同じレベルのLoss、BatchNorm2d入れると、最初画像が少し暗く見えますが、訓練は安定する

15. version 5, 真ん中にさらにCNNx１を追加する；CNNのkernelを小さくする.
> folder : 20180823-0545

結果：version 4 とほぼ同じ

16. version 6, 完全strideを避けるため -->　stride=2，kernel_size=3,padding=1; crop_size: 224 -> 129、細かい情報いらないから
> folder : 20180823-0755

結果：version 5とほぼ同じ

17. version 6, 16がもっといけそうなので、epoch数を長くする
> folder : 20180823-0908

結果：ゆっくり下がっていく。

以上の実験では、色認識は改善できませんでした。(12はわかりません。plotの画像は16枚しかなくて、何も言えない)

ということで、黒い文字（いらない情報）を除去して、色認識に集中してもらえるかな？

18. version 7, 文字なしのtarget、モデルはversion 6 と同じ。
> folder : 20180825-1617

結果：ちゃんと文字なしで出てきますが、色認識は改善しません。

19. version 7 にもしかして、num_featureが足りないではないか？num_feature : 32 --> 64
> folder : 20180826-0212

結果：色認識は改善しません。

---

結論：結局pix2codeでは、Resnet18(free)の方が予測精度一番いいモデルです。pix2codeの中にCNN Encoderを一緒に学習させるのは大事でしょう。AEを使う方では、モデルがより単純に作られていますので、予め学習させたとは言え、Test LossはResnet18(free)と比べるとちょっと高い。

結果：Testの175例の中に、以下のような間違いがします。（正解 --> 予測）
```
[   7/ 175] :
		btn-green  	-->	 btn-orange
		<unk>      	-->	 double    
		btn-orange 	-->	 btn-green 
		}          	-->	 double    
		<END>      	-->	 {         
		None       	-->	 small-title
		None       	-->	 ,         
		None       	-->	 text      
		None       	-->	 ,         
		None       	-->	 btn-red   
		None       	-->	 }         
		None       	-->	 }         
		None       	-->	 <END>     
[  30/ 175] :
		btn-orange 	-->	 btn-red   
		btn-green  	-->	 btn-orange
		btn-red    	-->	 btn-orange
		btn-red    	-->	 btn-orange
		btn-orange 	-->	 btn-red   
		quadruple  	-->	 <unk>     
		btn-green  	-->	 btn-orange
		quadruple  	-->	 }         
		{          	-->	 <END>     
		small-title 	-->	 None      
		,          	-->	 None      
		text       	-->	 None      
		,          	-->	 None      
		btn-green  	-->	 None      
		}          	-->	 None      
		quadruple  	-->	 None      
		{          	-->	 None      
		small-title 	-->	 None      
		,          	-->	 None      
		text       	-->	 None      
		,          	-->	 None      
		btn-red    	-->	 None      
		}          	-->	 None      
		quadruple  	-->	 None      
		{          	-->	 None      
		small-title 	-->	 None      
		,          	-->	 None      
		text       	-->	 None      
		,          	-->	 None      
		btn-red    	-->	 None      
		}          	-->	 None      
		}          	-->	 None      
		<END>      	-->	 None      
[ 101/ 175] :
		btn-green  	-->	 btn-orange
		btn-orange 	-->	 btn-green 
		btn-green  	-->	 btn-orange
		double     	-->	 <unk>     
		double     	-->	 }         
		{          	-->	 <END>     
		small-title 	-->	 None      
		,          	-->	 None      
		text       	-->	 None      
		,          	-->	 None      
		btn-orange 	-->	 None      
		}          	-->	 None      
		}          	-->	 None      
		<END>      	-->	 None      
[ 102/ 175] :
		quadruple  	-->	 double    
		quadruple  	-->	 double    
		quadruple  	-->	 }         
		{          	-->	 <END>     
		small-title 	-->	 None      
		,          	-->	 None      
		text       	-->	 None      
		,          	-->	 None      
		btn-green  	-->	 None      
		}          	-->	 None      
		quadruple  	-->	 None      
		{          	-->	 None      
		small-title 	-->	 None      
		,          	-->	 None      
		text       	-->	 None      
		,          	-->	 None      
		btn-green  	-->	 None      
		}          	-->	 None      
		}          	-->	 None      
		<END>      	-->	 None      
[ 105/ 175] :
		double     	-->	 quadruple 
		btn-green  	-->	 btn-orange
		double     	-->	 quadruple 
		btn-red    	-->	 btn-green 
		}          	-->	 quadruple 
		row        	-->	 {         
		{          	-->	 small-title
		quadruple  	-->	 ,         
		{          	-->	 text      
		small-title 	-->	 ,         
		,          	-->	 btn-red   
		text       	-->	 }         
		,          	-->	 quadruple 
		btn-green  	-->	 {         
		}          	-->	 small-title
		quadruple  	-->	 ,         
		{          	-->	 text      
		small-title 	-->	 ,         
		,          	-->	 btn-orange
		text       	-->	 }         
		,          	-->	 }         
		btn-green  	-->	 row       
		}          	-->	 {         
		quadruple  	-->	 <unk>     
		btn-orange 	-->	 btn-green 
		quadruple  	-->	 }         
		{          	-->	 <END>     
		small-title 	-->	 None      
		,          	-->	 None      
		text       	-->	 None      
		,          	-->	 None      
		btn-green  	-->	 None      
		}          	-->	 None      
		}          	-->	 None      
		<END>      	-->	 None      
[ 124/ 175] :
		btn-green  	-->	 btn-orange
		double     	-->	 <unk>     
		double     	-->	 }         
		{          	-->	 <END>     
		small-title 	-->	 None      
		,          	-->	 None      
		text       	-->	 None      
		,          	-->	 None      
		btn-orange 	-->	 None      
		}          	-->	 None      
		}          	-->	 None      
		<END>      	-->	 None 
```

ミス合計：6/175。その中、button色から間違うのは4例、layoutから間違うのは2例。

導出過程は[test_pix2code_training_result.ipynb](./test_pix2code_training_result.ipynb)をご覧ください。モデルは`20180805-123547`の300epochs訓練したResnet18(free)を使っています。
