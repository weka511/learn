# snarfed from https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1

from torch                  import device, relu
from torch.cuda             import is_available
from torch.nn               import Linear, Module, MSELoss
from torch.optim            import Adam
from torch.utils.data       import DataLoader
from torchvision.datasets   import MNIST
from torchvision.transforms import Compose, ToTensor

class AE(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = relu(activation)
        code = self.encoder_output_layer(activation)
        code = relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = relu(activation)
        return reconstructed

if __name__=='__main__':
    dev       = device("cuda" if is_available() else "cpu")
    model     = AE(input_shape=784).to(dev)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = MSELoss()

    transform = Compose([ToTensor()])

    train_dataset = MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True
    )

    test_dataset = MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    N = 10
    for epoch in range(N):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.view(-1, 784).to(dev)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, N, loss))
