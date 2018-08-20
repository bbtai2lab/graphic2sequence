#-----------------------------------------------------------------------------
# test very simple combination of CNN and FC to encode and decode bootstrap image
#-----------------------------------------------------------------------------

# %% import
import torch, torchvision
import os, PIL, glob, datetime, tensorboardX

# %%
class BootstrapDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, transform=None):

        self.data_path = data_path
        self.transform = transform

        self.raw_image_dirs = self.Prepare(data_path)

    def Prepare(self, data_path):

        raw_image_dirs= sorted( glob.glob(data_path+"*.png") )

        print("Created dataset of {:5d} items from\n{}".format(len(raw_image_dirs), data_path))


        return raw_image_dirs

    def __len__(self):

        return len(self.raw_image_dirs)

    def __getitem__(self, index):

        img_path = self.raw_image_dirs[index]

        image = PIL.Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image

# %% model
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

    def forward(self, x):                                                       # batch_s, in_channels, crop_size, crop_size
        out = self.encoder(x)                                                   # batch_s, num_feature, 56, 56
        out = out.view(out.size(0),-1)                                          # batch_s, num_feature*55*55
        out = self.latent(out)                                                  # batch_s, num_feature*55*55
        out = out.view(out.size(0), num_feature, 56, 56)                        # batch_s, num_feature, 56, 56
        out = self.decoder(out)                                                 # batch_s, in_channels, crop_size, crop_size

        return out

if __name__ == "__main__":

    # %% hyperparameters
    num_epoch = 100
    batch_size = 40
    crop_size = 224
    num_workers = 4
    num_latent = 128
    in_channels = 3
    num_feature = 50

    image_show_interval = 1
    model_save_interval = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on {}".format(device))

    # %% paths
    data_path = "../data/data_bootstrap"
    train_path = data_path + "/processed_data/data_train/"
    test_path = data_path + "/processed_data/data_test/"

    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((crop_size, crop_size)), # resize
            torchvision.transforms.ToTensor(),
            # See for : http://pytorch.org/docs/master/torchvision/models.html
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    train_dataset = BootstrapDataset(data_path=train_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    test_dataset = BootstrapDataset(data_path=test_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)

    # %% model
    autoencoder = AE(num_latent, in_channels, num_feature).to(device)
    #autoencoder = torch.load("./Logs/20180805_1220/AE_epoch_40.pkl").cuda()

    # %% criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr = 0.001)

    # %% training
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    writer = tensorboardX.SummaryWriter("./Logs/{}".format(time))
    info = "torch.nn.Conv2d(in_channels, num_feature, kernel_size=4, stride=2, padding=1) \n"
    info+= "torch.nn.Conv2d(num_feature, num_feature, kernel_size=4, stride=2, padding=1) \n"
    info+= "pretrained      : 40 epochs \n"
    info+= "batch_size      : {} \n".format(batch_size)
    info+= "crop_size       : {} \n".format(crop_size)
    info+= "num_epoch       : {} \n".format(num_epoch)
    info+= "num_latent      : {} \n".format(num_latent)
    info+= "num_feature     : {} \n".format(num_feature)
    info += "-"*60 + '\n'
    with open("./Logs/{}/info".format(time), 'a+') as file:
        file.write(info)

    autoencoder.train()
    for epoch in range(1, num_epoch+1):
        total_loss = 0.0
        for i, data_ in enumerate(train_loader, 1):                             # i starts from 1
            data = torch.autograd.Variable(data_.to(device))
            recon_data = autoencoder.forward(data)
            loss = criterion(recon_data, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        status = 'Epoch: [{:3d}/{:3d}] -->  Total loss: {:.4f} \n'.format(epoch, num_epoch, total_loss)
        print(status, end='')
        with open("./Logs/{}/info".format(time), 'a+') as file:
            file.write(status)

        writer.add_scalar('Train/Loss', total_loss, epoch)
        writer.add_scalar('Train/Progress', 100*(epoch)/num_epoch, epoch)

        if epoch % image_show_interval == 0:
            test_loss = 0.
            for data in test_loader:
                sample = torch.autograd.Variable(data.to(device))
                recon_data = autoencoder.forward(sample)
                loss = criterion(recon_data, sample)
                test_loss += loss.item()

            writer.add_scalar('Test/Loss', test_loss, epoch)
            writer.add_image("sample/origin", data[0,:,:,:], epoch)
            writer.add_image("sample/generated", recon_data[0,:,:,:], epoch)


        if epoch % model_save_interval == 0:
            torch.save(autoencoder, "./Logs/{}/AE_epoch_{}.pkl".format(time, epoch))
    writer.close()
