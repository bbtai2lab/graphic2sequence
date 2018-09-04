#-----------------------------------------------------------------------------
# Library of general purpose classes definition (version 1)
#-----------------------------------------------------------------------------

# %% Imports
import torch, torchvision
import PIL, glob

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
# list of ids --> string
#-----------------------------------------------------------------------------
def ids2str(id_list, vocab):
    word_list = list(map(lambda x: vocab.idx2word[x], id_list))
    return " ".join(word_list)
