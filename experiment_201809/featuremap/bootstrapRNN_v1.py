#-----------------------------------------------------------------------------
# train a RNN which is suitable for encoded array ->bootstrap captions
# - (version 1), batch_size=8
#   lr : 0.1
#   torch.optim.lr_scheduler.StepLR(optimizerD,step_size=200, gamma=0.1)
#   --> overfitting at test_loss~0.02
#-----------------------------------------------------------------------------

# %% import
import torch, torchvision
import os, random, PIL, glob, datetime, tensorboardX, math, argparse
from pix2code_lib import Vocabulary, BootstrapRNNDataset, collate_fn, TextViewer

#-----------------------------------------------------------------------------
# Decoder Model
#-----------------------------------------------------------------------------
# %%
class DecoderRNN(torch.nn.Module):

    def __init__(self, embed_size, hidden_size, num_vocab, num_layers, drop, bidirection):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()

        self.fc0 = torch.nn.Sequential(
            torch.nn.Linear(in_features=63, out_features=embed_size),
            #torch.nn.BatchNorm1d(embed_size),
        )
        self.embed = torch.nn.Embedding(num_vocab, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=drop, bidirectional=bidirection)
        self.fc = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, num_vocab),
                #torch.nn.BatchNorm1d(num_vocab),
                )

        self.embed_size = embed_size

    def forward(self, x, captions, lengths):
        """Decode image feature vectors and generates captions."""

        features = self.fc0(x)
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.fc(hiddens[0])

        return outputs

if __name__ == "__main__":

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
        "batch_size"    : 8,
        "num_workers"   : 4,
        "num_epochs"    : 4000,
        "seed"          : 1234,
        "embed_size"    : 128,
        "hidden_size"   : 256,
        "num_layers"    : 2,
        "LSTM_drop"     : 0.0,
        "bidirection"   : False,
        "isLimit"       : True if args.limit == "yes" else False,
        "crop_size"     : (224,224),
        "is_model_save" : False if args.limit == "yes" else True,
        "save_interval" : 100,
        "device"        : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "map_location"  : None if torch.cuda.is_available() else 'cpu',
        "model_path"    : "./Models/AE_precondition_release3/AE_precondition_state_dict_epoch_1000.pkl",
        "parent"        : None,
        "version"       : "bootstrapRNN_v1",
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

    if args.mode == "train":
        print("[Train Mode]")
        #-------------------------------------------------------------------------
        # define train and test dataset
        #-------------------------------------------------------------------------
        train_dataset = BootstrapRNNDataset(data_path=Ps["train_path"], vocab=vocab, isLimit=Ps["isLimit"])
        test_dataset = BootstrapRNNDataset(data_path=Ps["test_path"], vocab=vocab, isLimit=Ps["isLimit"])

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

        #  RNN decoder
        decoder = DecoderRNN(Ps["embed_size"], Ps["hidden_size"], Ps["vocab_size"], Ps["num_layers"], Ps["LSTM_drop"], Ps["bidirection"]).to(Ps["device"])

        #-- load pretrained parameters if necessary
        if Ps["parent"] is not None:
            cp = torch.load( f"./Models/{Ps['parent']}/checkpoint_best.pkl" , map_location=Ps["map_location"])
            decoder.load_state_dict( cp["decoder"] )

        #-------------------------------------------------------------------------
        # criterion and optimizer
        #-------------------------------------------------------------------------
        criterion = torch.nn.CrossEntropyLoss()
        optimizerD = torch.optim.SGD(params=decoder.parameters(), lr=1e-1, momentum=0.9, weight_decay=0)
        Ps["optimizerD"] = optimizerD.__repr__()

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
            info = "bootstrapRNN experiment with CPB"
            info+= "\nParameters:\n"
            for key in Ps.keys():
                info += f"{key:20s} : {Ps[key]} \n"
            info += "-"*60 + '\n'
            with open(f"{save_folder}/info.txt", 'a+') as file:
                file.write(info)

        #-------------------------------------------------------------------------
        # train and test
        #-------------------------------------------------------------------------
        acc_best = 0.98
        scheduleD = torch.optim.lr_scheduler.StepLR(optimizerD,step_size=200, gamma=0.1)
        optimizer = optimizerD
        schedule  = scheduleD

        for epoch in range(1, Ps["num_epochs"]+1):
            schedule.step()

            #-- train
            decoder.train()
            loss_total = 0.0
            acc_total = 0; acc_ref = 0
            for i, (arrays, captions, lengths) in enumerate(train_loader, 1):
                # Set mini-batch dataset
                arrays = arrays.to(Ps["device"]); arrays.require_grad=False
                captions = captions.to(Ps["device"]); captions.require_grad=False
                targets = torch.nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True).data
                # forward, backward and optimize
                outputs = decoder(arrays, captions, lengths)
                #-- accuracy
                acc_ref += targets.size(0)
                pred = torch.argmax(outputs, dim=1)
                for k in range(targets.size(0)):
                    if pred[k] == targets[k]:
                        acc_total += 1
                #-- loss
                loss = criterion(outputs, targets)
                loss_total += loss.item()
                decoder.zero_grad()
                loss.backward()
                optimizer.step()
            loss_train = loss_total / i
            acc_train = acc_total / acc_ref

            #-- validation
            decoder.train()
            loss_total = 0.0
            acc_total = 0; acc_ref = 0
            for i, (arrays, captions, lengths) in enumerate(test_loader, 1):
                # Set mini-batch dataset
                arrays = arrays.to(Ps["device"]); arrays.require_grad=False
                captions = captions.to(Ps["device"]); captions.require_grad=False
                targets = torch.nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True).data
                # forward, backward and optimize
                outputs = decoder(arrays, captions, lengths)
                #-- accuracy
                acc_ref += targets.size(0)
                pred = torch.argmax(outputs, dim=1)
                for k in range(targets.size(0)):
                    if pred[k] == targets[k]:
                        acc_total += 1
                #-- loss
                loss = criterion(outputs, targets)
                loss_total += loss.item()
            loss_test = loss_total / i
            acc_test = acc_total / acc_ref

            #-- print and log status
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            current_wd = optimizer.state_dict()['param_groups'][0]['weight_decay']

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

            #-- save model parameter
            if Ps["is_model_save"] and acc_test > acc_best:
                acc_best = acc_test
                print(f"!!! saving best test models at epoch: {epoch}")
                if not os.path.exists(f"./Models/{Ps['version']}/") : os.mkdir(f"./Models/{Ps['version']}/")
                checkpoint = {
                    "decoder" : decoder.state_dict(),
                    "Loss" : {"train" : loss_train, "test" : loss_test},
                    "Accuracy" : {"train" : acc_train, "test" : acc_test},
                }
                torch.save(checkpoint, os.path.join(f"./Models/{Ps['version']}/", f"checkpoint_best.pkl") )

            #-- save model every Ps["save_interval"] epochs
            if Ps["is_model_save"] and epoch % Ps["save_interval"] == 0:
                print(f"!!! checkpoint at epoch: {epoch}")
                if not os.path.exists(f"./Models/{Ps['version']}/") : os.mkdir(f"./Models/{Ps['version']}/")
                checkpoint = {
                    "decoder" : decoder.state_dict(),
                    "Loss" : {"train" : loss_train, "test" : loss_test},
                    "Accuracy" : {"train" : acc_train, "test" : acc_test},
                }
                torch.save(checkpoint, os.path.join(f"./Models/{Ps['version']}/", f"checkpoint_epoch_{epoch}.pkl") )

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
        decoder = DecoderRNN(Ps["embed_size"], Ps["hidden_size"], Ps["vocab_size"], Ps["num_layers"], Ps["LSTM_drop"], Ps["bidirection"]).to(Ps["device"])

        decoder.load_state_dict(torch.load( f"./Models/{Ps['version']}/decoder_state_dict_best.pkl" , map_location=Ps["map_location"]) )

        decoder.eval()
        #-------------------------------------------------------------------------
        # generate and save text
        #-------------------------------------------------------------------------
        tv = TextViewer(None, decoder, test_loader, Ps["device"], vocab)
        tv.generate_caption(f"./Logs/{Ps['version']}/caption_test.txt")
        tv.pick_mismatch(f"./Logs/{Ps['version']}/mismatch_test.txt")
        tv = TextViewer(None, decoder, dev_loader, Ps["device"], vocab)
        tv.generate_caption(f"./Logs/{Ps['version']}/caption_dev.txt")
        tv.pick_mismatch(f"./Logs/{Ps['version']}/mismatch_dev.txt")
