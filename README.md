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

**Result** : accuracy in Test dataset : 169/175 = 96.57%, including 4 mistakes caused by button color and 2 by layout.
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
# other files

- [./memo.md](./memo.md) : Log of experiments
- [./Library_v1.py](./Library_v1.py) : Vocabulary and Dataset class
- bootstrap.vocab : vocabulary for bootstrap


# AutoEncoding bootstrap images
---

Several experiments in [./AE_experiment](./AE_experiment) to see **whether we can solve the button color issue** by pretraining a CNN-Encoder by Auto-Encoding bootstrap images.

conclusion : Failed.

