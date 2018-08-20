#-----------------------------------------------------------------------------
# pix2code implementation with image caption method from
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
#
# - with resnet18 to be trained
# - optimizer : Adadelta
#-----------------------------------------------------------------------------

# %% Imports
import torch, torchvision, tensorboardX
import os, PIL, glob, datetime, numpy as np


#-----------------------------------------------------------------------------
# Vocabulary
#-----------------------------------------------------------------------------
# %%
class Vocabulary:
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.max_word_length = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            self.max_word_length = max(self.max_word_length, len(word))

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

#-----------------------------------------------------------------------------
# Dataset
#
# dataset is provided as `.gui` and `.pnf` pairs, so we create a custom Pytorch
# Dataset and Dataloader to stores captions in memory but loads images on-demands.
#-----------------------------------------------------------------------------
# %%
class Pix2codeDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, vocab, transform=None):

        self.data_path = data_path
        self.vocab = vocab
        self.transform = transform

        self.raw_image_dirs, self.raw_captions = self.Prepare(data_path)

    def Prepare(self, data_path):

        raw_image_dirs= sorted( glob.glob(data_path+"*.png") )

        raw_captions = []
        filenames_ = sorted( glob.glob(data_path+"*.gui") )
        #for i in range(10):
        #    print(raw_image_dirs[i].split('/')[-1][:-3]==filenames_[i].split('/')[-1][:-3])
        for filename in filenames_:
            with open(filename, 'r') as file:
                content = file.read()
            raw_captions.append(content)

        assert len(raw_captions)==len(raw_image_dirs), "bad dataset length."

        print("Created dataset of {:5d} items from\n{}".format(len(raw_captions), data_path))


        return raw_image_dirs, raw_captions

    def __len__(self):

        return len(self.raw_image_dirs)

    def __getitem__(self, index):

        img_path, raw_caption = self.raw_image_dirs[index], self.raw_captions[index]

        image = PIL.Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to list of vocab ID's
        caption = []
        caption.append(self.vocab('<START>'))

        # Remove newlines, separate words with spaces
        tokens = ' '.join(raw_caption.split())

        # Add space after each comma
        tokens = tokens.replace(',', ' ,')

        # Split into words
        tokens = tokens.split(' ')

        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<END>'))

        target = torch.Tensor(caption)

        return image, target

#-----------------------------------------------------------------------------
# Dataloader
#-----------------------------------------------------------------------------
# %%
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda pair: len(pair[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]                                    # List of caption lengths
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths


#-----------------------------------------------------------------------------
# Encoder Model
#-----------------------------------------------------------------------------
# %%
class EncoderCNN(torch.nn.Module):

    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN,self).__init__()

        resnet = torchvision.models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]                                  # except the last fc layer
        self.resnet = torch.nn.Sequential(*modules)
        self.fc = torch.nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = torch.nn.BatchNorm1d(embed_size,momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""

        # since resnet is not optimized for GUI screen shot, we have to train it
        # we can pre-train a new resnet by DCGAN or VAE
        features = self.resnet(images)
        features = features.reshape(features.size(0),-1)
        features = self.bn(self.fc(features))
        return features

#-----------------------------------------------------------------------------
# Decoder Model
#-----------------------------------------------------------------------------
# %%
class DecoderRNN(torch.nn.Module):

    def __init__(self, embed_size, hidden_size, num_vocab, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()

        self.embed = torch.nn.Embedding(num_vocab, embed_size)
        self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_vocab)

        self.embed_size = embed_size

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""

        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.fc(hiddens[0])

        return outputs

#-----------------------------------------------------------------------------
# criterion and optimizer
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# training
#-----------------------------------------------------------------------------
# %%
def train(encoder, decoder, train_loader, test_loader, criterion, optimizer,
          time, writer, num_epochs, log_interval, save_interval):

    for epoch in range(num_epochs):
        encoder.train(); decoder.train()
        print("-"*60)
        with open("./Logs/{}/info".format(time), "a+") as file:
            file.write("-"*60 + '\n')
        for i, (images, captions, lengths) in enumerate(train_loader):
            # Set mini-batch dataset
            images = torch.autograd.Variable( images.to(device) )
            captions = torch.autograd.Variable( captions.to(device) )
            targets = torch.nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True).data

            # forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                status = "Epoch [{:>4d}/{:<4d}] --> Loss : {:>4.4f}\tPerplexity: {:>5.4f}\n".format(
                                        epoch+1,num_epochs,loss.item(), torch.exp(loss).item())
                print(status,end='')
                with open("./Logs/{}/info".format(time), "a+") as file:
                    file.write(status)

                niter = epoch*len(train_loader)+i+1
                writer.add_scalar('Train/Loss', loss.item(), niter)
                writer.add_scalar('Train/Progress', 100*(epoch+1)/num_epochs, niter)

        # validation
        encoder.eval(); decoder.eval()
        loss_test_total = 0.0
        for j, (images_test, captions_test, lengths_test) in enumerate(test_loader):

            images_test = torch.autograd.Variable( images_test.to(device) )
            captions_test = torch.autograd.Variable( captions_test.to(device) )
            targets_test = torch.nn.utils.rnn.pack_padded_sequence(captions_test, lengths_test, batch_first=True).data
            features_test = encoder(images_test)
            outputs_test = decoder(features_test, captions_test, lengths_test)
            loss_test = criterion(outputs_test, targets_test)
            loss_test_total += loss_test.item()
        loss_test_total /= j+1
        writer.add_scalar('Test/Loss', loss_test_total, epoch)

        if (epoch+1) % save_interval == 0:
            print('!!! saving models at epoch: ' + str(epoch+1))
            torch.save(encoder, os.path.join("./Logs/{}/".format(time), "encoder_epoch{}.pkl".format(epoch)) )
            torch.save(decoder, os.path.join("./Logs/{}/".format(time), "decoder_epoch{}.pkl".format(epoch)) )


if __name__ == "__main__":

    # %% Hyperparameters
    # these 4 parameters need to be changed if training in AWS
    batch_size = 50
    num_workers = 4
    log_interval = 7
    num_epochs = 800


    embed_size = 256
    hidden_size = 512
    num_layers = 1
    crop_size = 224                                                                 # required by resnet, do not change
    save_interval = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("train on {}".format(device))

    # %% Paths
    data_path = "../data/data_bootstrap/"
    train_path = data_path + "processed_data/data_train/"
    test_path = data_path + "processed_data/data_test/"
    vocab_path = data_path + "bootstrap.vocab"
    # model_save_path = "./model_saved/"

    # %% read vocabulary content from file
    with open(vocab_path, 'r') as file:
        content = file.read()
    words = content.split(' '); del content
    len(words)

    # %% vocabulary object
    vocab = Vocabulary()
    for word in words:
        vocab.add_word(word)
    vocab.add_word(' ')
    vocab.add_word('<unk>') # if we find an unknown word
    vocab_size = len(vocab)
    print("total length of vocab is : {}".format(vocab_size))

    # %%  Transform to modify images for pre-trained ResNet base and train_dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((crop_size, crop_size)), # Match resnet size
        torchvision.transforms.ToTensor(),
        # See for : http://pytorch.org/docs/master/torchvision/models.html
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = Pix2codeDataset(data_path=train_path, vocab=vocab, transform=transform)
    test_dataset = Pix2codeDataset(data_path=test_path, vocab=vocab, transform=transform)

    # %% data_loader with user defined collate_fn
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    # %% models
    #  CNN encoder
    encoder = EncoderCNN(embed_size).to(device)

    #  RNN decoder
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)

    # %% criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters())
    #optimizer = torch.optim.Adam(params, lr=0.001)
    optimizer = torch.optim.Adadelta(params=params,lr=1.0)

    # %% tensorboardX
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print("time = {}".format(time))
    writer = tensorboardX.SummaryWriter("./Logs/{}".format(time))

    # information
    infofile = open("./Logs/{}/info".format(time), "a+")
    info = "CNN model    : resnet18, fixed=False \n"
    info+= "what accuracy can we acheive if we have many many epochs. \n"
    info+= "num_epochs   : {} \n".format(num_epochs)
    info+= "batch_size   : {} \n".format(batch_size)
    info+= "log_interval : {} \n".format(log_interval)
    info+= "optimizer    : Adadelta \n"
    with open("./Logs/{}/info".format(time), "a+") as file:
        file.write(info)


    # %% training
    train(encoder, decoder, train_loader, test_loader, criterion, optimizer,
          time, writer, num_epochs, log_interval, save_interval)

    print("Training Terminated.")
    writer.close()



    # %% send email to me
    #SendEmail.sendGmail2Me()
