#-----------------------------------------------------------------------------
# pix2code training script
# - (version 1), with pretrained AE_precondition
#   1. add BatchNorm1d
#   --> loss curve decrease faster, but still overfitting
#   2. embed_size : 256 -> 128, hidden_size : 512 -> 256 | let's try to underfit the model
#   --> overfitting
#   3. embed_size : 128 -> 64 , hidden_size : 256 -> 128 | let's try to underfit the model
#   --> slightly overfitting
#   4. embed_size : 64  -> 32 , hidden_size : 128 -> 64  | let's try to underfit the model
#   --> slightly overfitting, volatile training loss also (due to the small capacity of our model)
#   5. batch_size : 64 -> 128
#   --> not yet overfitting, volatile training loss also, num_epochs=800 is not enough
#   6. embed_size : 32  -> 64 , hidden_size : 64 -> 128, num_epochs : 800 -> 1600
#   --> overfitting
#   7. add dropout 0.5 right before encoder.feature
#   --> strong overfitting
#   8. dropout : 0.5 -> 0.2
#   --> very volatile validation error. overfit at last
#   9. remove dropout before encoder.feature, add dropout after encoder.feature, 0.5
#   --> overfitting at last
#   10. add dropout before encoder.feature, 0.3, batch_size : 128 -> 64
#   --> overfits, 0.3 is too large
#   11. dropout : 0.3 -> 0.15
#   --> overfitting
#   12. remove the first dropout. Adadelta -> Adam
#   --> loss decrease faster, however, overfit
#   13. add batchnorm after decoder.fc
#   --> loss goes lower, anyway, overfit occur at the same epoch
#   14. reduce hidden_size : 128 -> 96
#   --> overfitting improved a little bit
#   15. reduce hidden_size : 96 -> 64
#   --> overfitting improved a little bit
#   16. weight_decay : 1e-4
#   --> looks good! no overfitting!
#   17. num_epochs : 1600 -> 3200
#   --> absolutely underfitting! looks not bad!
#   18. let's increase the model capacity. hidden_size : 64 -> 128
#   --> training loss gets better; no help on test loss, slight overfitting.
#   19. hidden_size : 128 -> 64; num_layers : 1 -> 2
#   --> worse than num_layers=1, overfit anyway
#   20. num_layers : 2 -> 1, weight_decay : 1e-4 -> 2e-4
#   --> num_epochs not enough
#   21. num_epochs : 3200 -> 4800, add current lr and current weight_decay
#   --> not bad. no overfitting, test loss -> 0.026, but very volatile. should we decrease learning rate?
# - (version 2), Adam -> SGD with 1cycle policy, try a wide range of learning rate to decide the optimized learning rate
#   1. lr : 1e-4
#   2. lr : 2e-4, num_epochs : 4800 -> 1600
#   3. lr : 5e-4
#   4. lr : 1e-3
#   5. lr : 2e-3
#   6. lr : 5e-3
#   --> overcome the bottleneck @ epoch1580
#   7. lr : 1e-2
#   8. lr : 5e-2
#   --> looks not bad, test loss < 0.04
#   9. lr : 5e-2, num_epochs : 1600 -> 3200
#   --> overfitting @ epoch 2000 with test loss < 0.03
#   10. lr : 1e-1
#   --> overfitting @ epoch 1500 with test loss ~ 0.03
#   11. lr : 2e-2, num_epochs : 3200 -> 4800
#   --> overfitting @ epoch 3000 with test loss ~ 0.22
#   12. schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1600,2800], gamma=0.1, last_epoch=-1); lr=0.1
#   --> lr being 0.001 is too small?
#   13. remove the dropout inside encoder.feature
#       schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,200,500], gamma=0.2, last_epoch=-1)
#       lr = 0.1
#       num_epochs : 4800 -> 2400
#   --> not good.
#   14. schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)
#       lr = 0.1
#   --> overfitting, test loss saturate at ~ 0.02
#   15. weight_decay : 0.0 -> 2e-4; lr : 1e-1 -> 2e-2
#   16. weight_decay : 2e-4 -> 3e-4; add linear decay to lr amplitude; num_epochs : 2400 -> 4800
#   --> looks not good
#   17. exponentialLR : gamma : 0.9988
#   18. lr : 2e-2 -> 5e-3, embed_size : 64 -> 128, hidden_size : 64 -> 128
#   19. exponential decay; num_epochs : 4800 -> 10000
#   20. num_epochs : 10000 -> 20000
#       save best model and optimizer
#       hidden_size : 128 -> 256
#       num_layers : 1 -> 2
#       batch_size : 64 -> 16
#   --> test loss saturates ~ 0.11. train loss also saturates. underfitting?
#   21. lr_min : 1e-4 -> 6e-4
#   --> best test loss with 0.091. both test loss and train loss saturate with difference being ~ 0.002
#       12. train with new bootstrap.vocab
#           shuffle twice
#           1. continue v2.21.12 with lr : 6e-4 -> 5e-3
#               batch_size : 16 -> 64
#               no schedule
#               weight_decay : 3e-4 -> 0
#           --> not good.
#           2. with parent v2.21.12, lr : 1e-3, cos schedule
#               batch_size : 64 -> 16
#               fix decoder (forgot to set LSTM_drop to 0.0). train all encoder
#           --> most layout mismatch and many btn-color mismatch is improved.
#               this tells us that the main issue we have to overcome is how to train EncoderCNN perfectly.
#           3. with parent v2.21.12.2, lr : 1e-2 ~ 1e-1, cos schedule, T_max : 100 -> 50
#               acc_best : 0.995 -> 0.998
#               LSTM_drop : 0.1 -> 0.0
#           --> not bad. but the improvement of v2.21.12.2 is tiny.

#   next : add BatchNorm2d right before encoder.feature?
#   next : real time learning rate adjustment using jupyter notebook?
#   next : orthogonal init for LSTM?
#-----------------------------------------------------------------------------


# %% import
import torch, torchvision
import os, random, PIL, glob, datetime, tensorboardX, math, argparse
from pix2code_lib import Vocabulary, Pix2codeDataset, collate_fn, TextViewer
from AE_precondition_release3 import AE_precondition

#-----------------------------------------------------------------------------
# Encoder Model
#-----------------------------------------------------------------------------
# %%
class EncoderCNN(torch.nn.Module):

    def __init__(self, num_feature, embed_size, model_path, map_location):
        """Load the pretrained Autoencoder and replace last fc layer in model_latent."""
        super(EncoderCNN,self).__init__()

        #AE = torch.load(model_path, map_location=map_location)
        AE = AE_precondition(num_feature)
        AE.load_state_dict(torch.load(model_path,map_location=map_location))
        self.conv_color = AE.conv_color
        self.downSample = AE.downSample
        self.conv_block1 = AE.conv_block1
        self.conv_block2 = AE.conv_block2
        self.conv_block3 = AE.conv_block3

        self.feature = torch.nn.Sequential(
            torch.nn.Linear(in_features=7*32, out_features=embed_size),
            torch.nn.BatchNorm1d(embed_size),
            #torch.nn.Dropout(0.5),
            )

    def forward(self, x):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            x = self.conv_color(x)
            x = self.downSample(x)

        out = []
        for i in range(7):
            xx = x[:,i,:,:].unsqueeze(1)
            xx = self.conv_block3( self.conv_block2( self.conv_block1(xx) ) )
            out.append(xx)
        out = torch.cat(out,dim=1)

        features = out.view(out.size(0),-1)
        features = self.feature(features)

        return features

#-----------------------------------------------------------------------------
# Decoder Model
#-----------------------------------------------------------------------------
# %%
class DecoderRNN(torch.nn.Module):

    def __init__(self, embed_size, hidden_size, num_vocab, num_layers, drop, bidirection):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()

        self.embed = torch.nn.Embedding(num_vocab, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=drop, bidirectional=bidirection)
        self.fc = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, num_vocab),
                torch.nn.BatchNorm1d(num_vocab),
                )

        self.embed_size = embed_size

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""

        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.fc(hiddens[0])

        return outputs


if __name__=="__main__":

    #-------------------------------------------------------------------------
    # train or test
    #-------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='training or generate caption, train/test', default="train")
    parser.add_argument('--limit', type=str, help='whether to limit the size of dataset, yes/no', default="no")
    args = parser.parse_args()
    assert args.mode in ("train","test"), "wrong --mode argument."
    assert args.limit in ("yes","no"), "wrong --limit argument."

    #-------------------------------------------------------------------------
    # Hyperparameters
    #-------------------------------------------------------------------------
    Ps = {
        "batch_size"    : 16,
        "num_workers"   : 4,
        "num_epochs"    : 10000,
        "seed"          : 1234,
        "num_feature"   : 8,
        "embed_size"    : 128,
        "hidden_size"   : 256,
        "num_layers"    : 2,
        "LSTM_drop"     : 0.0,
        "bidirection"   : False,
        "isLimit"       : True if args.limit == "yes" else False,
        "crop_size"     : 96,
        "is_model_save" : False if args.limit == "yes" else True,
        "text_interval" : 1000,
        "device"        : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "map_location"  : None if torch.cuda.is_available() else 'cpu',
        "model_path"    : "./Models/AE_precondition_release3/AE_precondition_state_dict_epoch_1000.pkl",
        "parent"        : "pix2code_v2.21.12.2",
        "version"       : "pix2code_v2.21.12.3",
        "root_path"     : "../data/data_bootstrap/"
    }
    Ps["train_path"] = Ps["root_path"] + "processed_data/data_train/"
    Ps["test_path"] = Ps["root_path"] + "processed_data/data_test/"
    Ps["dev_path"] = Ps["root_path"] + "processed_data/data_dev/"
    Ps["vocab_path"] = Ps["root_path"] + "bootstrap.vocab"

    #-------------------------------------------------------------------------
    # create vocabulary object
    #-------------------------------------------------------------------------
    with open(Ps["vocab_path"], 'r') as file:
        content = file.read()
    words = content.split(' '); del content
    random.seed(Ps["seed"])
    for i in range(2):
        random.shuffle(words)

    vocab = Vocabulary()
    for word in words:
        vocab.add_word(word)
    vocab.add_word(' ')
    vocab.add_word('<unk>') # if we find an unknown word
    Ps["vocab_size"] = len(vocab)
    #print(f"total length of vocab is : {Ps['vocab_size']}")

    #-------------------------------------------------------------------------
    # define transformation for training images
    #-------------------------------------------------------------------------
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((Ps["crop_size"], Ps["crop_size"])),
        torchvision.transforms.ToTensor(),
    ])

    if args.mode == "train":
        print("[Train Mode]")
        #-------------------------------------------------------------------------
        # define train and test dataset
        #-------------------------------------------------------------------------
        train_dataset = Pix2codeDataset(data_path=Ps["train_path"], vocab=vocab, transform=transform, isLimit=Ps["isLimit"])
        test_dataset = Pix2codeDataset(data_path=Ps["test_path"], vocab=vocab, transform=transform, isLimit=Ps["isLimit"])

        #-------------------------------------------------------------------------
        # data_loader with user defined collate_fn
        #-------------------------------------------------------------------------
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=Ps["batch_size"],
                                    shuffle=True, num_workers=Ps["num_workers"], collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=Ps["batch_size"],
                                    shuffle=True, num_workers=Ps["num_workers"], collate_fn=collate_fn)

        #-------------------------------------------------------------------------
        # models
        #-------------------------------------------------------------------------
        torch.manual_seed(Ps["seed"])
        #  CNN encoder
        encoder = EncoderCNN(Ps["num_feature"], Ps["embed_size"], Ps["model_path"], Ps["map_location"]).to(Ps["device"])

        #  RNN decoder
        decoder = DecoderRNN(Ps["embed_size"], Ps["hidden_size"], Ps["vocab_size"], Ps["num_layers"], Ps["LSTM_drop"], Ps["bidirection"]).to(Ps["device"])

        #-- load pretrained parameters if necessary
        if Ps["parent"] is not None:
            encoder.load_state_dict(torch.load( f"./Models/{Ps['parent']}/encoder_state_dict_best.pkl" , map_location=Ps["map_location"]) )
            decoder.load_state_dict(torch.load( f"./Models/{Ps['parent']}/decoder_state_dict_best.pkl" , map_location=Ps["map_location"]) )

        # fix the weights in decoder
        for params in decoder.parameters():
            params.require_grad = False

        #-------------------------------------------------------------------------
        # criterion and optimizer
        #-------------------------------------------------------------------------
        criterion = torch.nn.CrossEntropyLoss()
        #params = list(decoder.parameters())# + list(encoder.feature.parameters())
        #params = list(decoder.parameters()) + list(encoder.parameters())
        params = encoder.parameters()
        #optimizer = torch.optim.Adam(params=params, lr=1e-3, weight_decay=2e-4)
        optimizer = torch.optim.SGD(params=params, lr=1e-1, momentum=0.9, weight_decay=0.)
        Ps["optimizer"] = optimizer.__repr__()

        if not Ps["isLimit"]:
            #-------------------------------------------------------------------------
            # tensorboardX
            #-------------------------------------------------------------------------
            save_folder = f"./Logs/{Ps['version']}"
            assert not os.path.exists(save_folder), f"{save_folder} already exist."
            writer = tensorboardX.SummaryWriter(save_folder)

            #-------------------------------------------------------------------------
            # heads of info.txt
            #-------------------------------------------------------------------------
            info = "pix2code experiment with CPB"
            info+= "\nParameters:\n"
            for key in Ps.keys():
                info += f"{key:20s} : {Ps[key]} \n"
            info += "-"*60 + '\n'
            with open(f"{save_folder}/info.txt", 'a+') as file:
                file.write(info)

        #-------------------------------------------------------------------------
        # train and test
        #-------------------------------------------------------------------------
        acc_best = 0.998
        #schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,200,500], gamma=0.2, last_epoch=-1)
        schedule  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-2)
        #schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9988, last_epoch=-1)
        for epoch in range(1, Ps["num_epochs"]+1):
            schedule.step()
            #for param_group in optimizer.param_groups:
            #    param_group['lr'] = 0.01 * math.exp(-40*epoch/(Ps["num_epochs"]+1))+6e-4


            #-- train
            encoder.train(); decoder.train()
            loss_total = 0.0
            acc_total = 0; acc_ref = 0
            for i, (images, captions, lengths) in enumerate(train_loader, 1):
                # Set mini-batch dataset
                images = images.to(Ps["device"]); images.require_grad=False
                captions = captions.to(Ps["device"]); captions.require_grad=False
                targets = torch.nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True).data
                # forward, backward and optimize
                features = encoder(images)
                outputs = decoder(features, captions, lengths)
                #-- accuracy
                acc_ref += targets.size(0)
                pred = torch.argmax(outputs, dim=1)
                for k in range(targets.size(0)):
                    if pred[k] == targets[k]:
                        acc_total += 1
                #-- loss
                loss = criterion(outputs, targets)
                loss_total += loss.item()
                decoder.zero_grad(); encoder.zero_grad()
                loss.backward()
                optimizer.step()
            loss_train = loss_total / i
            acc_train = acc_total / acc_ref

            #-- validation
            encoder.eval(); decoder.eval()
            loss_total = 0.0
            acc_total = 0; acc_ref = 0
            for j, (images, captions, lengths) in enumerate(test_loader, 1):
                # Set mini-batch dataset
                images = images.to(Ps["device"]); images.require_grad=False
                captions = captions.to(Ps["device"]); captions.require_grad=False
                targets = torch.nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True).data
                # forward
                features = encoder(images)
                outputs = decoder(features, captions, lengths)
                #-- accuracy
                acc_ref += targets.size(0)
                pred = torch.argmax(outputs, dim=1)
                for k in range(targets.size(0)):
                    if pred[k] == targets[k]:
                        acc_total += 1
                #-- loss
                loss = criterion(outputs, targets)
                loss_total += loss.item()
            loss_test = loss_total / j
            acc_test = acc_total / acc_ref

            #-- print and log status
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            current_wd = optimizer.state_dict()['param_groups'][0]['weight_decay']

            #print(optimizer.state_dict())

            status = f"Epoch [{epoch:>4d}/{Ps['num_epochs']:<4d}] --> Train/Loss : {loss_train:>2.4f}\tTest/Loss : {loss_test:>2.4f}\tOptimizer/lr : {current_lr:>2.5f}\tOptimizer/wd : {current_wd:>2.5f}\n"
            status += f"|-- Train/Acc : {acc_train*100:2.4f}%\tTest/Acc : {acc_test*100:2.4f}%\n"
            print(status,end='')
            if not Ps["isLimit"]:
                with open(f"{save_folder}/info.txt", 'a+') as file:
                    file.write(status)

                #-- tensorboardX writer
                writer.add_scalar('Loss/Train', loss_train, epoch)
                writer.add_scalar('Loss/Test', loss_test, epoch)
                writer.add_scalar('Loss/Test-Train', loss_test-loss_train, epoch)
                writer.add_scalar('Accuracy/Train', acc_train*100, epoch)
                writer.add_scalar('Accuracy/Test', acc_test*100, epoch)
                writer.add_scalar('Optimizer/lr', current_lr, epoch)
                writer.add_scalar('Optimizer/wd', current_wd, epoch)

            #-- generate sample text
            if epoch % Ps["text_interval"] == 0:
                pass

            #-- save model parameter
            if Ps["is_model_save"] and acc_test > acc_best:
                acc_best = acc_test
                print(f"!!! saving models at epoch: {epoch}")
                if not os.path.exists(f"./Models/{Ps['version']}/") : os.mkdir(f"./Models/{Ps['version']}/")
                torch.save(encoder.state_dict(), os.path.join(f"./Models/{Ps['version']}/", f"encoder_state_dict_best.pkl") )
                torch.save(decoder.state_dict(), os.path.join(f"./Models/{Ps['version']}/", f"decoder_state_dict_best.pkl") )
                torch.save(optimizer.state_dict(), os.path.join(f"./Models/{Ps['version']}/", f"optimizer_state_dict_best.pkl") )

        writer.close()

    #-------------------------------------------------------------------------
    # test mode
    #-------------------------------------------------------------------------
    if args.mode == "test":

        print("[Test Mode]")
        #-------------------------------------------------------------------------
        # dataset and dataloader
        #-------------------------------------------------------------------------
        test_dataset = Pix2codeDataset(data_path=Ps["test_path"], vocab=vocab, transform=transform, isLimit=Ps["isLimit"])
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
        dev_dataset = Pix2codeDataset(data_path=Ps["dev_path"], vocab=vocab, transform=transform, isLimit=Ps["isLimit"])
        dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
        #-------------------------------------------------------------------------
        # define models and load parameters
        #-------------------------------------------------------------------------
        encoder = EncoderCNN(Ps["num_feature"], Ps["embed_size"], Ps["model_path"], Ps["map_location"]).to(Ps["device"])
        decoder = DecoderRNN(Ps["embed_size"], Ps["hidden_size"], Ps["vocab_size"], Ps["num_layers"], Ps["LSTM_drop"], Ps["bidirection"]).to(Ps["device"])

        encoder.load_state_dict(torch.load( f"./Models/{Ps['version']}/encoder_state_dict_best.pkl" , map_location=Ps["map_location"]) )
        decoder.load_state_dict(torch.load( f"./Models/{Ps['version']}/decoder_state_dict_best.pkl" , map_location=Ps["map_location"]) )

        encoder.eval(); decoder.eval()
        #-------------------------------------------------------------------------
        # generate and save text
        #-------------------------------------------------------------------------
        tv = TextViewer(encoder, decoder, test_loader, Ps["device"], vocab)
        tv.generate_caption(f"./Logs/{Ps['version']}/caption_test.txt")
        tv.pick_mismatch(f"./Logs/{Ps['version']}/mismatch_test.txt")
        tv = TextViewer(encoder, decoder, dev_loader, Ps["device"], vocab)
        tv.generate_caption(f"./Logs/{Ps['version']}/caption_dev.txt")
        tv.pick_mismatch(f"./Logs/{Ps['version']}/mismatch_dev.txt")
