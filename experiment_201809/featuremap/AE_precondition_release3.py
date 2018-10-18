#-----------------------------------------------------------------------------
# 让CNN最大程度抽取bootstrap特征图的实验
# - (version 1), 最大化特征图之间的`-torch.nn.BCELoss`来使其识别颜色。
#   1. BootstrapDataset_preload, limit the size of dataset for the sake of saving time
#   2. num_color: 6 -> 9
#   3. Sigmoid -> Hardtanh
#   4. add one more lay to perform linear combination between different feature maps
#   5. remove limit
#   6. remove the 2nd CNN block; num_color: 9 -> 5
#   7. the 1st CNN block: 1x1 -> 3x3
#   8. in Maximize_featuremaps, sum += k * torch.sum(tensor_); k=0.003
#       --> epoch 105, able to recognize orange
#   9. k: 0.003 -> 0.001
# - (version 2), add keyword isSame to BootstrapDataset_preload; color splitted auto-encoder.
#   --> good! color match; position match. but the resolution of recon_data is a little bit low
#   1. "nearest" -> "bilinear" in torch.nn.Upsample
#       --> gradient vanished
#   2. try again
#       --> found some fake signal caused by "bilinear".
#   3. leave only the last "bilinear"; num_epoch: 400 -> 600
#       --> loss even lower than v2.2, but recon_data still looks bad and remains some fake signal
#   4. variable batch_size
#       --> failed
# - (release), same as v2.0, num_feature : 8 -> 6, seed : 1234
#   1. num_feature : 6 -> 8
#   2. num_feature : 8 -> 12, test -> dev
#   3. num_feature : 12 -> 8, dev -> test, seed : 1234 -> 2234
#-----------------------------------------------------------------------------

# %% import
import torch, torchvision
import os, random, PIL, glob, datetime, tensorboardX

# %%
class BootstrapDataset_preload(torch.utils.data.Dataset):

    def __init__(self, data_path, transform=None, isSame=False):

        self.data_path = data_path
        self.transform = transform
        self.isSame = isSame

        self.Prepare(data_path)

    def Prepare(self, data_path):


        raw_target_dirs= sorted( glob.glob(data_path+"*.png") )
        #limit = 100 if len(raw_target_dirs) > 300 else 20; raw_target_dirs = raw_target_dirs[:limit]

        if self.isSame:
            raw_image_dirs = raw_target_dirs
        else:
            raw_image_dirs = list()
            for item in raw_target_dirs:
                raw_image_dirs.append(item.replace("data_bootstrap_notext","data_bootstrap"))

        print("Created input dataset of {:5d} items from\n{}".format(len(raw_image_dirs), data_path.replace("data_bootstrap_notext", "data_bootstrap")))
        print("Created target dataset of {:5d} items from\n{}".format(len(raw_target_dirs), data_path))

        image_list = []
        for dir in raw_image_dirs:
            img = PIL.Image.open(dir).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            image_list.append(img)

        if self.isSame:
            target_list = image_list
        else:
            target_list = []
            for dir in raw_target_dirs:
                img = PIL.Image.open(dir).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                target_list.append(img)

        self.raw_image_dirs = raw_image_dirs
        self.raw_target_dirs = raw_target_dirs
        self.image_list = image_list
        self.target_list = target_list

    def __len__(self):

        return len(self.raw_image_dirs)

    def __getitem__(self, index):

        image  = self.image_list[index]
        if self.isSame:
            target = image
        else:
            target = self.target_list[index]

        return image, target

class BootstrapDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, transform=None):

        self.data_path = data_path
        self.transform = transform

        self.raw_image_dirs, self.raw_target_dirs = self.Prepare(data_path)

    def Prepare(self, data_path):

        raw_target_dirs= sorted( glob.glob(data_path+"*.png") )
        raw_image_dirs = list()
        for item in raw_target_dirs:
            raw_image_dirs.append(item.replace("data_bootstrap_notext","data_bootstrap"))

        print("Created input dataset of {:5d} items from\n{}".format(len(raw_image_dirs), data_path.replace("data_bootstrap_notext", "data_bootstrap")))
        print("Created target dataset of {:5d} items from\n{}".format(len(raw_target_dirs), data_path))


        return raw_image_dirs, raw_target_dirs

    def __len__(self):

        return len(self.raw_image_dirs)

    def __getitem__(self, index):

        img_path = self.raw_image_dirs[index]
        tag_path = self.raw_target_dirs[index]

        image  = PIL.Image.open(img_path).convert('RGB')
        target = PIL.Image.open(tag_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            target = self.transform(target)

        return image, target

# %% model
class AE_precondition(torch.nn.Module):

    def __init__(self, num_feature):
        super(AE_precondition, self).__init__()

        self.conv_color = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 7, 1, bias=True),
                        #torch.nn.BatchNorm2d(7),
                        #torch.nn.Hardtanh(min_val=0, max_val=1),
                        torch.nn.ReLU(),

                        torch.nn.Conv2d(7, 7, 1, bias=True),
                        #torch.nn.BatchNorm2d(7),
                        torch.nn.Sigmoid(),
        )
        self.downSample = torch.nn.MaxPool2d(3, stride=3) # 96 -> 32

        num_features = (num_feature, 2*num_feature, 4*num_feature)

        self.conv_block1 = torch.nn.Sequential( # 32 -> 16
                        torch.nn.Conv2d(1, num_features[0], kernel_size=3, stride=1, padding=1),
                        #torch.nn.BatchNorm2d(num_features[0]),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(2, stride=2),
        )
        self.conv_block2 = torch.nn.Sequential( # 16 -> 8
                        torch.nn.Conv2d(num_features[0], num_features[1], kernel_size=3, stride=1, padding=1),
                        #torch.nn.BatchNorm2d(num_features[1]),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(2, stride=2),
        )
        self.conv_block3 = torch.nn.Sequential( # 8 -> 1
                        torch.nn.Conv2d(num_features[1], num_features[2], kernel_size=8, stride=1, padding=0),
                        #torch.nn.BatchNorm2d(num_features[2]),
                        torch.nn.ReLU(),
        )
        self.deconv_block3 = torch.nn.Sequential( # 1 -> 8
                        torch.nn.ConvTranspose2d(num_features[2], num_features[1], kernel_size=8, stride=1, padding=0),
                        torch.nn.ReLU(),
        )
        self.deconv_block2 = torch.nn.Sequential( # 8 -> 16
                        torch.nn.Upsample(scale_factor=2, mode='nearest'),
                        torch.nn.ConvTranspose2d(num_features[1], num_features[0], kernel_size=3, stride=1, padding=1),
                        torch.nn.ReLU(),
        )
        self.deconv_block1 = torch.nn.Sequential( # 16 -> 32
                        torch.nn.Upsample(scale_factor=2, mode='nearest'),
                        torch.nn.ConvTranspose2d(num_features[0], 1, kernel_size=3, stride=1, padding=1),
                        torch.nn.ReLU(),
        )
        self.upSample = torch.nn.Upsample(scale_factor=3, mode='nearest') # 32 -> 96

        self.deconv_color = torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(7, 7, 1, bias=True),
                        torch.nn.ReLU(),

                        torch.nn.ConvTranspose2d(7, 3, 1, bias=True),
                        torch.nn.Sigmoid(),
        )

        self.blocks = [self.conv_block1,
                        self.conv_block2,
                        self.conv_block3,
                        self.deconv_block3,
                        self.deconv_block2,
                        self.deconv_block1]


    def forward(self, x):

        with torch.no_grad():
            x = self.conv_color(x)
        x = self.downSample(x)

        out = []
        for i in range(7):
            xx = x[:,i,:,:].unsqueeze(1)
            for block in self.blocks:
                xx = block(xx)
            out.append(xx)
        out = torch.cat(out,dim=1)
        out = self.upSample(out)
        out = self.deconv_color(out)

        return out



if __name__ == "__main__":

    # %% hyperparameters
    num_epoch = 1000
    batch_size = 36
    crop_size = 96
    num_workers = 4
    num_feature = 8
    seed = 2234
    lr = 1e-3
    momentum = 0.9
    weight_decay = 0
    version = "AE_precondition_release3"

    image_show_interval = 40
    model_save_interval = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on {}".format(device))

    # %% paths
    data_path = "../data/data_bootstrap"
    train_path = data_path + "/processed_data/data_train/"
    test_path = data_path + "/processed_data/data_test/"
    #dev_path = data_path + "/processed_data/data_dev/"

    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((crop_size, crop_size)), # resize
            torchvision.transforms.ToTensor(),
        ])
    train_dataset = BootstrapDataset_preload(data_path=train_path, transform=transform, isSame=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    test_dataset = BootstrapDataset_preload(data_path=test_path, transform=transform, isSame=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=36,shuffle=True, num_workers=num_workers)
    #dev_dataset = BootstrapDataset_preload(data_path=dev_path, transform=transform)
    #dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset, batch_size=64,shuffle=True, num_workers=num_workers)

    # %% model
    torch.manual_seed(seed)
    model = AE_precondition(num_feature).to(device)
    model.conv_color.load_state_dict(torch.load("./Models/conv_color_state_dict.pkl"))

    # %% criterion and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)

    # %% training
    #time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    folder = "./Logs/"
    assert not os.path.exists("{}{}".format(folder, version)), "folder {} already exist.".format("{}{}".format(folder,version))
    writer = tensorboardX.SummaryWriter("{}{}".format(folder, version))
    info = "preconditioned AutoEncoder for bootstrap images ({})".format(version)
    info+= "\n"
    info+= "batch_size      : {} \n".format(batch_size)
    info+= "crop_size       : {} \n".format(crop_size)
    info+= "num_epoch       : {} \n".format(num_epoch)
    info+= "num_feature     : {} \n".format(num_feature)
    info += "-"*60 + '\n'
    with open("{}{}/info".format(folder, version), 'a+') as file:
        file.write(info)

    for epoch in range(1, num_epoch+1):

        # if epoch == num_epoch//3:
        #     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2*batch_size,shuffle=True, num_workers=num_workers)
        # elif epoch == num_epoch//3*2:
        #     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=4*batch_size,shuffle=True, num_workers=num_workers)

        #-- train
        model.train()

        train_loss = 0.0
        for i, (data, target) in enumerate(train_loader):
            data = torch.autograd.Variable(data.to(device))
            recon_data = model.forward(data)
            loss = criterion(recon_data, target.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= i+1

        #-- validation
        model.eval()

        test_loss = 0.
        for j, (data,target) in enumerate(test_loader):
            data = torch.autograd.Variable(data.to(device))
            recon_data = model.forward(data)
            loss = criterion(recon_data, target.to(device))

            test_loss += loss.item()
        test_loss /= j+1

        status = 'Epoch: [{:3d}/{:3d}] -->  Train/Loss: {:.6f}\tTest/Loss: {:.6f} \n'.format(epoch, num_epoch, train_loss, test_loss)
        print(status, end='')
        with open("{}{}/info".format(folder, version), 'a+') as file:
            file.write(status)

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)

        #-- show feature maps
        if epoch % image_show_interval == 0:
            for data, target in test_loader:
                data = torch.autograd.Variable(data.to(device))
                recon_data = model.forward(data)
                break
            writer.add_image("Sample/original", torchvision.utils.make_grid(target[:,:,:,:], nrow=6, padding=3), epoch)
            writer.add_image("Sample/generated", torchvision.utils.make_grid(recon_data[:,:,:,:], nrow=6, padding=3), epoch)


        if epoch % model_save_interval == 0:
            if not os.path.exists(f"./Models/{version}/") : os.mkdir(f"./Models/{version}/")
            torch.save(model.state_dict(), "./Models/{}/AE_precondition_state_dict_epoch_{}.pkl".format(version, epoch))
    writer.close()
