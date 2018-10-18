#-----------------------------------------------------------------------------
# Library for pix2code experiment
#-----------------------------------------------------------------------------

# %% Imports
import torch, torchvision
import PIL, glob, random, numpy as np

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
# dataset is provided as `.gui` and `.png` pairs, so we create a custom Pytorch
# Dataset and Dataloader to store images and captions in memory.
#-----------------------------------------------------------------------------
# %%
class Pix2codeDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, vocab, transform=None, isLimit=False):
        r""" """
        self.data_path = data_path
        self.vocab = vocab
        self.transform = transform
        self.isLimit=isLimit

        self.Prepare()

    def Prepare(self):
        r""" """
        data_path = self.data_path

        #-- prepare images
        image_dirs= sorted( glob.glob(data_path+"*.png") )
        if self.isLimit:
            limit = 100 if len(image_dirs) > 300 else 20
            image_dirs = image_dirs[:limit]

        image_list = []
        for dir in image_dirs:
            img = PIL.Image.open(dir).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            image_list.append(img)

        #-- prepare captions
        caption_dirs = [dir.replace(".png", ".gui") for dir in image_dirs]

        caption_list = []
        for dir in caption_dirs:
            with open(dir, 'r') as file:
                content = file.read()
            caption_list.append(content)

        assert len(caption_list)==len(image_list), "bad dataset length."

        print(f"Created dataset of {len(image_list):5d} items from\n{data_path}")

        #-- assign class attribute
        self.image_list = image_list
        self.caption_list = caption_list


    def __len__(self):
        r""" necessary for __getitem__ """
        return len(self.caption_list)

    def __getitem__(self, index):
        r""" """
        image, raw_caption = self.image_list[index], self.caption_list[index]

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

        return image.type(torch.FloatTensor), target.type(torch.FloatTensor)

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
# list of ids --> string
#-----------------------------------------------------------------------------
def ids2str(id_list, vocab):
    r""" """
    word_list = list(map(lambda x: vocab.idx2word[x], id_list))
    return " ".join(word_list)

#-----------------------------------------------------------------------------
# generate caption
#-----------------------------------------------------------------------------
class TextViewer:

    def __init__(self, encoder, decoder, data_loader, device, vocab):
        r""" """
        self.encoder = encoder
        self.decoder = decoder
        self.data_loader = data_loader
        self.device = device
        self.vocab = vocab

    def generate_caption(self, save_path=None):
        r""" """
        prediction_list = []
        caption_list = []
        for j, (images, captions, lengths) in enumerate(self.data_loader, 1):
            images = images.to(self.device)
            if self.encoder is not None:
                features = self.encoder(images)
            else:
                features = self.decoder.fc0(images)

            for b in range(captions.size(0)): # seperate batch
                caption_list.append(captions.numpy()[b,:].tolist())   # there is an extra batch dimension

                sampled_ids = []
                inputs = features[b,:].unsqueeze(0).unsqueeze(1)
                states = None
                for i in range(100):  # predict 100 words at most
                    #-- predict
                    hiddens, states = self.decoder.lstm(inputs, states)
                    outputs = self.decoder.fc(hiddens.squeeze(1))
                    _, predicted = outputs.max(1)

                    #-- store prediction
                    sampled_ids.append(predicted.item())

                    # if predicts <END>, break
                    if sampled_ids[-1] == self.vocab.word2idx['<END>']:
                        break

                    # prediction as the next input
                    inputs = self.decoder.embed(predicted)
                    inputs = inputs.unsqueeze(1)
                prediction_list.append(sampled_ids)

        # with open("./Logs/{}/info".format(time), "a+") as file:
        #     file.write(ids2str(caption_list[-1], vocab)+"\n")
        #     file.write(ids2str(prediction_list[-1], vocab)+"\n")
        self.prediction_list = prediction_list
        self.caption_list = caption_list

        if save_path is not None:
            with open(save_path, "w+") as file:
                for i in range(len(prediction_list)):
                    file.write(ids2str(caption_list[i], self.vocab)+"\n")
                    file.write(ids2str(prediction_list[i], self.vocab)+"\n")
                    file.write('_'*60+'\n')
            print(f"{save_path} has been saved.")

    def pick_mismatch(self, save_path):
        r""" """
        prediction_list = self.prediction_list
        caption_list = self.caption_list
        vocab = self.vocab

        info = ""
        for i in range(len(prediction_list)):
            info_ = "[{:4d}/{:4d}] :\n".format( i+1,len(prediction_list) )

            count = 0
            if len(prediction_list[i]) >= len(caption_list[i]):
                for n in range(len(prediction_list[i])):
                    if n >= len(caption_list[i]):
                        info_ += "\t\t{:10s} \t-->\t {:10s}\n".format("None", vocab.idx2word[prediction_list[i][n]])
                        count += 1
                    elif prediction_list[i][n] != caption_list[i][n]:
                        info_ += "\t\t{:10s} \t-->\t {:10s}\n".format(vocab.idx2word[caption_list[i][n]], vocab.idx2word[prediction_list[i][n]])
                        count += 1
                    else:
                        pass

            if len(prediction_list[i]) < len(caption_list[i]):
                for n in range(len(caption_list[i])):
                    if n >= len(prediction_list[i]):
                        info_ += "\t\t{:10s} \t-->\t {:10s}\n".format(vocab.idx2word[caption_list[i][n]], "None")
                        count += 1
                    elif prediction_list[i][n] != caption_list[i][n]:
                        info_ += "\t\t{:10s} \t-->\t {:10s}\n".format(vocab.idx2word[caption_list[i][n]], vocab.idx2word[prediction_list[i][n]])
                        count += 1
                    else:
                        pass

            if count > 0:
                info += info_
        self.mismatch = info

        if save_path is not None:
            with open(save_path, "w+") as file:
                file.write(info)
            print(f"{save_path} has been saved.")

#-----------------------------------------------------------------------------
# Dataset
#
# pair of encoded .gui array and captions
#-----------------------------------------------------------------------------
class BootstrapRNNDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, vocab, isLimit=False):
        r""" """
        self.data_path = data_path
        self.isLimit=isLimit
        self.vocab=vocab

        self.Prepare()

    def Prepare(self):
        r""" """
        data_path = self.data_path
        #-- prepare captions
        caption_dirs= sorted( glob.glob(data_path+"*.gui") )
        if self.isLimit:
            limit = 100 if len(caption_dirs) > 300 else 20
            caption_dirs = caption_dirs[:limit]
        caption_list = []
        for dir in caption_dirs:
            with open(dir, 'r') as file:
                content = file.read()
            caption_list.append(content)
        #-- prepare arrays
        eg = EncoderGui()
        eg.files2captions(caption_dirs)
        eg.captions2layouts()
        eg.encode_layouts()

        assert eg.arrays.shape[0]==len(caption_list), "bad dataset length."
        print(f"Created dataset of {len(caption_list):5d} items from\n{data_path}")

        #-- assign class attribute
        self.caption_list = caption_list
        self.eg = eg

    def __len__(self):
        r""" necessary for __getitem__ """
        return self.eg.arrays.shape[0]

    def __getitem__(self, index):
        r""" """
        #-- arrays
        array = torch.from_numpy( self.eg.arrays[index,:] )
        #-- captions
        raw_caption = self.caption_list[index]

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

        return array.type(torch.FloatTensor), target.type(torch.FloatTensor)


#-----------------------------------------------------------------------------
# Dataset
#
# pair of .png image and encoded .gui array
#-----------------------------------------------------------------------------
class BootstrapCNNDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, transform=None, isLimit=False):
        r""" """
        self.data_path = data_path
        self.transform = transform
        self.isLimit=isLimit

        self.Prepare()

    def Prepare(self):
        r""" """
        data_path = self.data_path
        #-- prepare images
        image_dirs= sorted( glob.glob(data_path+"*.png") )
        if self.isLimit:
            limit = 100 if len(image_dirs) > 300 else 20
            image_dirs = image_dirs[:limit]

        image_list = []
        for dir in image_dirs:
            img = PIL.Image.open(dir).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            image_list.append(img)

        #-- prepare captions
        caption_dirs = [dir.replace(".png", ".gui") for dir in image_dirs]
        eg = EncoderGui()
        eg.files2captions(caption_dirs)
        eg.captions2layouts()
        eg.encode_layouts()

        assert eg.arrays.shape[0]==len(image_list), "bad dataset length."
        print(f"Created dataset of {len(image_list):5d} items from\n{data_path}")

        #-- assign class attribute
        self.image_list = image_list
        self.eg = eg

    def __len__(self):
        r""" necessary for __getitem__ """
        return self.eg.arrays.shape[0]

    def __getitem__(self, index):
        r""" """
        image = self.image_list[index]
        target = self.eg.arrays[index,:]
        target = torch.from_numpy(target)

        return image.type(torch.FloatTensor), target.type(torch.FloatTensor)

def CrossValidation(dataset1, dataset2, seed, ratio=0.9, N=3):
    r"""Croos Validation : combine, shuffle and then split two datasets """

    image_list = dataset1.image_list + dataset2.image_list
    arrays = np.append(dataset1.arrays, dataset2.arrays, axis=0)
    assert len(image_list)==arrays.shape[0], "bad length."
    L = arrays.shape[0]

    random.seed(seed)
    order = [i for i in range(L)]
    for i in range(N): random.shuffle(order)
    boundary = int(L*ratio)
    order_train = order[:boundary]; order_test = order[boundary:]
    image_train = [image_list[i] for i in order_train]
    image_test = [image_list[i] for i in order_test]
    array_train = arrays[order_train,:]
    array_test = arrays[order_test,:]

    dataset1.image_list = image_train; dataset2.image_list = image_test
    dataset1.arrays = array_train; dataset2.arrays = array_test


#-----------------------------------------------------------------------------
# encode captions in .gui files
#-----------------------------------------------------------------------------
class EncoderGui:

    def __init__(self, path=None):

        if path is not None:
            self.files = glob.glob(path+"/*.gui")
            print(f"found {len(self.files)} .gui files")

        self.pos = {
            "empty"          : 0,
            "btn-inactive"   : 1,
            "btn-active"     : 2,
            "btn-green"      : 1,
            "btn-red"        : 2,
            "btn-orange"     : 3,
        }

    def encode_layouts(self):

        print(f"converting {len(self.layouts)} layouts to arrays")
        layouts = self.layouts
        arrays = np.empty((len(layouts),63), dtype=np.float32)
        for i in range(arrays.shape[0]):
            arrays[i,:] = self.encode_one_layout(layouts[i])
        self.arrays = arrays

    def encode_one_layout(self, layout):

        res = np.zeros(63, dtype=np.float32)
        #-- encode btn-inactive and btn-active
        for j in range(5):
            res[ j*3 + self.pos[layout[0][j]] ] = 1.
        #-- encode btn-color
        for i in range(1,4):
            for j in range(4):
                res[15+(i-1)*16+j*4 + self.pos[layout[i][j]]] = 1.

        return res

    def files2captions(self, files=None):

        if files is None:
            files = self.files

        print(f"converting {len(files)} .gui files to captions")
        captions = []
        for file in files:
            with open(file, 'r') as f:
                captions.append(f.readlines())
        self.captions = captions

    def captions2layouts(self):

        print(f"converting {len(self.captions)} captions to layouts")
        layouts = []
        for caption in self.captions:
            layouts.append( self.extract_words_from_caption(caption) )

        self.layouts = layouts

    def extract_words_from_caption(self, caption):

        layout = []
        #-- btn-inactite and btn-active
        layout.append(caption[1][:-1].split(", "))
        assert len( layout[0] ) <= 5, "no more than 5 btn-(in)active"
        #-- padding btn-inactive and btn-active
        for i in range(5-len( layout[0] )):
            layout[0].append("empty")

        #-- find rows
        rows = []
        for i, line in enumerate(caption):
            if line[:3]=="row":
                rows.append(i)
        rows.append(-1)
        nrows = len(rows)-1
        assert nrows <= 3, "no more than 3 rows."

        #-- find btn in each row
        for i in range(nrows):
            count = 0
            btns = ["empty","empty","empty","empty"]
            for line in caption[rows[i]:rows[i+1]]:
                count_inline = 0
                for btn in ("btn-red","btn-green","btn-orange"):
                    if btn in line:
                        btns[count] = btn
                        count_inline += 1
                assert count_inline <= 1, "no more than one button in one line."
                if count_inline > 0:
                    count += 1
            layout.append(btns)

        #-- padding rows
        for i in range(3-nrows):
            layout.append( ["empty","empty","empty","empty"] )

        #-- make them tuple
        layout_tuple = []
        for item in layout:
            layout_tuple.append( tuple(item) )
        return tuple(layout_tuple)

def get_prediction(output):
    r"""
    output : (63,)
    """
    res = np.zeros(output.size(0),dtype=np.float32)
    posi = [0,3,6,9,12,15,19,23,27,31,35,39,43,47,51,55,59,63]
    for i in range(len(posi)-1):
        k = torch.argmax( output[posi[i]:posi[i+1]] )
        res[posi[i]+k] = 1.
    return res

if __name__ == "__main__":
    eg = EncoderGui("../data/data_bootstrap/processed_data/data_train/")
    eg.files2captions()
    eg.captions2layouts()
    eg.encode_layouts()

    i = 1200
    print(eg.files[i])
    print(eg.layouts[i])
    print(eg.arrays[i,:])
