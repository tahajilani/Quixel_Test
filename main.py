from customDataloader import *
from cnnModel import *

transformed_dataset = customDataset(root_dir='train-data/',
                                           transform=transforms.Compose([
                                               RandomCrop(254),
                                               ToTensor(),
                                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                                           ]))


dataloader = DataLoader(transformed_dataset, batch_size=1,
                        shuffle=True, num_workers=0)


# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch = \
            sample_batched
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

# if you are using Windows, uncomment the next line and indent the for loop.
# you might need to go back and change "num_workers" to 0.

# if __name__ == '__main__':
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched.size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break





batch_size = 4
learning_rate = 1e-3
num_epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = VAE().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


"""
Training the network for a given number of epochs
The loss after every epoch is printed
# """
# for epoch in range(num_epochs):
#     for idx, data in enumerate(dataloader):
#         imgs = data
#         # print(imgs)
#         imgs = imgs.to(device)

#         # Feeding a batch of images into the network to obtain the output image, mu, and logVar
#         out, mu, logVar = net(imgs)
#                 # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
#         # print(out.shape)
#         kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
#         # print(out)
#         loss = F.mse_loss(out, imgs, reduction='mean') + kl_divergence #using this cuz recommended in a post
#         # Backpropagation based on the loss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print('Epoch {}: Loss {}: KL{}'.format(epoch, loss,kl_divergence))



# torch.save(net, 'model.pth')
# print('model saved')

net=torch.load('model.pth')

import matplotlib.pyplot as plt
import numpy as np
import random

net.eval()

with torch.no_grad():
    for data in random.sample(list(dataloader), 1):
        imgs = data
        img = np.transpose(imgs[0].numpy(), [1,2,0])
        plt.subplot(121)
        plt.imshow(np.squeeze(img))
        out, mu, logVAR = net(imgs)
        outimg = np.transpose(out[0].numpy(), [1,2,0])
        plt.subplot(122)
        plt.imshow(np.squeeze(outimg))
        break


out=out.squeeze(0).numpy()
out=out[0]
# print(out.shape())
# out=int(out*255)
print(out.max())
cv2.imshow(out)
cv2.waitkey(-1)