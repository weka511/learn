# snarfed from https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1

from matplotlib.pyplot      import figure, show
from argparse               import ArgumentParser
from torch                  import device, relu
from torch.cuda             import is_available
from torch.nn               import Linear, Module, MSELoss
from torch.optim            import Adam
from torch.utils.data       import DataLoader
from torchvision.datasets   import MNIST
from torchvision.transforms import Compose, ToTensor

class AutoEncoder(Module):
    def __init__(self, input_shape=784):
        super().__init__()
        self.encoder_hidden_layer = Linear(in_features  = input_shape,
                                           out_features = 128)
        self.encoder_output_layer = Linear(in_features  = 128,
                                           out_features = 128)
        self.decoder_hidden_layer = Linear(in_features  = 128,
                                           out_features = 128)
        self.decoder_output_layer = Linear(in_features  = 128,
                                           out_features = input_shape)

    def forward(self, features):
        activation    = self.encoder_hidden_layer(features)
        activation    = relu(activation)
        code          = self.encoder_output_layer(activation)
        code          = relu(code)
        activation    = self.decoder_hidden_layer(code)
        activation    = relu(activation)
        activation    = self.decoder_output_layer(activation)
        reconstructed = relu(activation)
        return reconstructed

if __name__=='__main__':
    parser        = ArgumentParser('Autoencoder')
    parser.add_argument('--N',
                        type    = int,
                        default = 25,
                        help    = 'Number of Epochs')
    args          = parser.parse_args()
    dev           = device("cuda" if is_available() else "cpu")
    model         = AutoEncoder(input_shape=784).to(dev)
    optimizer     = Adam(model.parameters(), lr=1e-3)
    criterion     = MSELoss()
    transform     = Compose([ToTensor()])

    train_dataset = MNIST(root="~/torch_datasets",
                          train     = True,
                          transform = transform,
                          download  = True)
    test_dataset  = MNIST(root="~/torch_datasets",
                          train     = False,
                          transform = transform,
                          download  = True)
    train_loader  = DataLoader(train_dataset,
                               batch_size  = 128,
                               shuffle     = True,
                               num_workers = 4,
                               pin_memory  = True)
    test_loader   = DataLoader(test_dataset,
                               batch_size  = 32,
                               shuffle     = False,
                               num_workers = 4)
    Losses        = []

    for epoch in range(args.N):
        loss = 0
        for batch_features, _ in train_loader:
            batch_features = batch_features.view(-1, 784).to(dev)
            optimizer.zero_grad()
            outputs = model(batch_features)
            train_loss = criterion(outputs, batch_features)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        Losses.append(loss / len(train_loader))
        print(f'epoch : {epoch+1}/{args.N}, loss = {Losses[-1]:.6f}')

    fig = figure(figsize=(10,10))
    ax  = fig.subplots()
    ax.plot(Losses)
    ax.set_ylim(bottom=0)
    show()
