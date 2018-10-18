#-----------------------------------------------------------------------------
# 寻找能够独立激活各种单色的权重
#-----------------------------------------------------------------------------

import torch, torchvision
import numpy as np, glob, matplotlib.pyplot as plt
from PIL import Image

def init_2d_color(Rs,Gs,Bs, n):
    res = np.empty((3,n), dtype=np.float32)
    res[0,:] = np.linspace(Rs[0], Rs[1], n)
    res[1,:] = np.linspace(Gs[0], Gs[1], n)
    res[2,:] = np.linspace(Bs[0], Bs[1], n)
    return res

class ColorType:

    def __init__(self, nRand=64):

        self.nRand = nRand

        self.green_data = init_2d_color((0.242,0.451),(0.561,0.761),(0.243,0.451), nRand)
        self.red_data = init_2d_color((0.757,0.847),(0.184,0.322),(0.169,0.306), nRand)
        self.orange_data = init_2d_color((0.922,0.941),(0.580,0.675),(0.098,0.298), nRand)

    def get_one_color(self, k):

        assert k < self.nRand, "bad length."

        color0 = np.array([[0.2,0.2,0.961,1.0,0,0,0],
                        [0.2,0.478,0.961,1.0,0,0,0],
                        [0.2,0.718,0.961,1.0,0,0,0]])
        color0[:,4] = self.green_data[:,k]
        color0[:,5] = self.red_data[:,k]
        color0[:,6] = self.orange_data[:,k]

        return color0.reshape(3,7,1)

    def get_batch(self, batch_size=64):

        res = np.empty((batch_size,3,7,1), np.float32)
        for k in range(batch_size):
            res[k,:,:,:] = self.get_one_color(k)

        return torch.from_numpy( res )

    def get_target(self, batch_size=64):

        res = np.empty((batch_size,7,7,1), np.float32)
        res[:,:,:,0] = np.eye(7)

        return torch.from_numpy( res )




if __name__ == "__main__":

    # colors = torch.tensor([[0.20,0.20,0.20],
    #                        [0.20,0.48,0.72],
    #                        [1.00,1.00,1.00],
    #                        [0.96,0.96,0.96],
    #                        [0.31,0.66,0.31],
    #                        [0.81,0.27,0.25],
    #                        [0.93,0.63,0.2]])
    # colors.transpose_(0,1)
    # colors = colors.view(1,3,7,1)
    #weights = torch.ones((7,3)) * 0.3; weights = torch.autograd.Variable(weights, requires_grad=True)
    #bias = torch.zeros((7,1)); bias = torch.autograd.Variable(bias, requires_grad=True)

    # target = torch.tensor([[1,0,0,0,0,0,0],
    #                        [0,1,0,0,0,0,0],
    #                        [0,0,1,0,0,0,0],
    #                        [0,0,0,1,0,0,0],
    #                        [0,0,0,0,1,0,0],
    #                        [0,0,0,0,0,1,0],
    #                        [0,0,0,0,0,0,1]]).float()
    # target = target.view(1,7,7,1)

    model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 7, 1, bias=True),
                #torch.nn.BatchNorm2d(7),
                #torch.nn.Hardtanh(min_val=0, max_val=1),
                torch.nn.ReLU(),

                torch.nn.Conv2d(7, 7, 1, bias=True),
                #torch.nn.BatchNorm2d(7),
                torch.nn.Sigmoid(),

                )

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD( model.parameters(), lr = 0.1, momentum=0.9, weight_decay=0.0)
    #optimizer = torch.optim.Adam( model.parameters() )


    if False:
        fig = plt.figure()
        color_type = ColorType()
        colors = color_type.get_batch()
        target = color_type.get_target()

        num_epochs = 6001
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1.0)
        for epoch in range(1, num_epochs):
            scheduler.step()

            model.train()
            out = model(colors)
            loss = criterion(out, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if epoch % 10 == 0:
                print("{:3.4f} % -> train Loss : {:2.4f}".format(100*epoch/num_epochs, loss.item()), end='\n')
                b = np.random.randint(0,64)
                plt.imshow(out.detach().numpy()[b,:,:,0], cmap="gray", vmin=0.0, vmax=1)
                plt.pause(0.01)
        #
        #print(model[0].weight.view(-1), model[0].bias)
        #print(model[3].weight.view(-1), model[3].bias)
        #print(out[0,:,:,0])

        # save model dict
        torch.save(model.state_dict(), "./conv_color_state_dict.pkl")
    else:
        model.load_state_dict(torch.load("./conv_color_state_dict.pkl"))

    # load bootstrap image and test output
    filepath = "../data/data_bootstrap/processed_data/data_test/"
    file = glob.glob(filepath+"*.png")[0]
    image = Image.open(file).convert("RGB")
    crop_size = 96
    transform = torchvision.transforms.Compose([
         torchvision.transforms.Resize((crop_size, crop_size)), # resize
         torchvision.transforms.ToTensor(),
         ])
    image = transform(image).unsqueeze(0)
    out = model(image).detach().numpy()

    fig2, ax2 = plt.subplots(2,4, figsize=(12,6),dpi=100)
    for i in range(2):
        for j in range(4):
            n = i*4+j
            if n > 6:
                ax2[i,j].imshow(image.numpy()[0,:,:,:].transpose((1,2,0)))
            else:
                ax2[i,j].imshow(out[0,n,:,:], cmap="gray", vmin=0.0, vmax=1)


    plt.show()
