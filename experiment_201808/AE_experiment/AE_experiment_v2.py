#-----------------------------------------------------------------------------
# test very simple combination of CNN and FC to encode and decode bootstrap image
#   - 单纯地增加 num_latent 并不能带来多少好处，反而 Test/Loss 变得非常不稳定， 这也许是过拟合的征兆。
#     所以问题还是出现在CNN的网络结构上
#   - 改变了网络结构，假如(1x1,s=1)和(6x6,s=4)CNN；可变的batch_size。(v1, failed)
#       --> (1x1,s=1)的加入使得图片失去了区别RGB的能力，这与我预料中的不符。
#   - 不管怎样，取消(1x1,s=1)，并且，仔细想想，好像AE并不是一定得有中间的latent层，
#     尝试latent层去掉直接连接encoder与decoder？并且摒弃bias。(v2, success)
#       --> 取消了latent层的信息压缩，理所当然地能够复原图像，因为承载信息的神经元量级上基本没减少（只减少了一半）。
#           这也是导致Test/Loss下降很快但是不稳定的原因？试试减少num_feature
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

        self.num_feature = num_feature

        self.encoder = torch.nn.Sequential(

                    torch.nn.Conv2d(in_channels, num_feature, kernel_size=4, stride=2, padding=1, bias=False), # 112
                    torch.nn.BatchNorm2d(num_feature),
                    torch.nn.LeakyReLU(0.2, inplace=True),

                    torch.nn.Conv2d(num_feature, num_feature*2, kernel_size=6, stride=4, padding=1, bias=False), # 28
                    torch.nn.BatchNorm2d(num_feature*2),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    )
        self.latent = torch.nn.Sequential(
                    torch.nn.Linear(num_feature*2*28*28, num_latent),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                    torch.nn.Linear(num_latent, num_feature*2*28*28),
                    )

        self.decoder = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(num_feature*2, num_feature, kernel_size=6, stride=4, padding=1, bias=False), # 112
                    torch.nn.LeakyReLU(0.2, inplace=True),

                    torch.nn.ConvTranspose2d(num_feature, in_channels, kernel_size=4, stride=2, padding=1, bias=False), # 224
                    torch.nn.Sigmoid(),
                    )

    def forward(self, x):                                                       # batch_s, in_channels, crop_size, crop_size
        out = self.encoder(x)                                                   # batch_s, num_feature*2, 28, 28
        # out = out.view(out.size(0),-1)                                          # batch_s, num_feature*2*28*28
        # out = self.latent(out)                                                  # batch_s, num_feature*2*28*28
        # out = out.view(out.size(0), self.num_feature*2, 28, 28)                 # batch_s, num_feature*2, 28, 28
        out = self.decoder(out)                                                 # batch_s, in_channels, crop_size, crop_size

        return out

if __name__ == "__main__":

    # %% hyperparameters
    num_epoch = 150
    batch_size = 100
    crop_size = 224
    num_workers = 4
    num_latent = 128
    in_channels = 3
    num_feature = 60

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
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16,shuffle=True, num_workers=num_workers)

    # %% model
    autoencoder = AE(num_latent, in_channels, num_feature).to(device)

    # %% criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr = 0.001)

    # %% training
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    writer = tensorboardX.SummaryWriter("./Logs/{}".format(time))
    info = """
    AE generative model on bootstrap images (version 2)
    torch.nn.Conv2d(in_channels, num_feature, kernel_size=4, stride=2, padding=1, bias=False), # 112
    torch.nn.Conv2d(num_feature, num_feature*2, kernel_size=6, stride=4, padding=1, bias=False), # 28
    variable batch_size \n
    """
    info+= "batch_size      : {} \n".format(batch_size)
    info+= "crop_size       : {} \n".format(crop_size)
    info+= "num_epoch       : {} \n".format(num_epoch)
    info+= "num_latent      : {} \n".format(num_latent)
    info+= "num_feature     : {} \n".format(num_feature)
    info+= "optimizer       : Adam\n"
    info += "-"*60 + '\n'
    with open("./Logs/{}/info".format(time), 'a+') as file:
        file.write(info)

    for epoch in range(1, num_epoch+1):
        if epoch == 50:
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=40,shuffle=True, num_workers=num_workers)
        elif epoch == 100:
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=8,shuffle=True, num_workers=num_workers)

        autoencoder.train()
        total_loss = 0.0
        for i, data_ in enumerate(train_loader, 1):                             # i starts from 1
            data = torch.autograd.Variable(data_.to(device))
            recon_data = autoencoder.forward(data)
            loss = criterion(recon_data, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        total_loss /= i

        autoencoder.eval()
        test_loss = 0.
        for j, data in enumerate(test_loader,1):
            sample = torch.autograd.Variable(data.to(device))
            recon_data = autoencoder.forward(sample)
            loss = criterion(recon_data, sample)

            test_loss += loss.item()
        test_loss /= j

        status = 'Epoch: [{:3d}/{:3d}] -->  Train/Loss: {:.4f}\tTest/Loss: {:.4f} \n'.format(epoch, num_epoch, total_loss, test_loss)
        print(status, end='')
        with open("./Logs/{}/info".format(time), 'a+') as file:
            file.write(status)

        writer.add_scalar('Train/Loss', total_loss, epoch)
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Train/Progress', 100*epoch/num_epoch, epoch)

        if epoch % image_show_interval == 0:
            for data in test_loader:
                sample = torch.autograd.Variable(data.to(device))
                recon_data = autoencoder.forward(sample)
                break
            writer.add_image("sample/origin", torchvision.utils.make_grid(data[:,:,:,:], nrow=4, padding=20), epoch)
            writer.add_image("sample/generated", torchvision.utils.make_grid(recon_data[:,:,:,:], nrow=4, padding=20), epoch)


        if epoch % model_save_interval == 0:
            torch.save(autoencoder, "./Logs/{}/AE_epoch_{}.pkl".format(time, epoch))
    writer.close()
