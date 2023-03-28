import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils.util import FolderDataset, set_seeds

class MLP(nn.Module):

    def __init__(self, img_dim=784, num_classes=10):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(img_dim, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.log_softmax(self.net(x), dim=1)

    def pred(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.softmax(self.net(x), dim=1)

class LogReg(nn.Module):

    def __init__(self, img_dim=784, num_classes=10):
        super(LogReg, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(784, num_classes),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.log_softmax(self.net(x), dim=1)

    def pred(self, x):
        x = x.reshape(x.shape[0], -1)
        return F.softmax(self.net(x), dim=1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.model = torch.nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1152, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.model(x)

def train_cnn(loader1, loader2, loader3, loader4, device, max_epochs=50):
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    objective = nn.CrossEntropyLoss()

    return train_model(loader1, loader2, loader3, loader4, device, model, optimizer, objective, max_epochs)

def train_mlp(loader1, loader2, loader3, loader4, device, max_epochs=50):
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    objective = lambda x, y: F.nll_loss(x, y)

    return train_model(loader1, loader2, loader3, loader4, device, model, optimizer, objective, max_epochs)

def train_log_reg(loader1, loader2, loader3, loader4, device, max_epochs=50):
    model = LogReg().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    objective = lambda x, y: F.nll_loss(x, y)

    return train_model(loader1, loader2, loader3, loader4, device, model, optimizer, objective, max_epochs)

def train_model(loader1, loader2, loader3, loader4, device, model, optimizer, objective, max_epochs):
    best_acc = best_train_acc = best_test_acc = 0.

    for _ in range(max_epochs):
        for _, (train_x, train_y) in enumerate(loader1):
            x = train_x.to(device).to(torch.float32) * 2. - 1.
            y = train_y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = objective(outputs, y)
            loss.backward()
            optimizer.step()

        model.eval()
        acc, _ = compute_acc(model, loader2, device)
        if acc > best_acc:
            best_acc = acc
            best_train_acc, _ = compute_acc(model, loader3, device)
            best_test_acc, _ = compute_acc(model, loader4, device)
        model.train()

    return best_train_acc, best_test_acc

def compute_acc(model, loader, device):
    test_loss = 0
    correct = 0
    outputs = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = data.to(torch.float32)
            data = data * 2. - 1.
            output = model(data)
            output = nn.functional.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().cpu().item()
            outputs.append(output)
    preds = torch.cat(outputs, dim=0)
    test_loss /= loader.dataset.__len__()
    acc = correct / loader.dataset.__len__()
    return acc, preds

def train_all_classifiers(train_set_loader, test_set_loader, dataset_name, device, batch_size):
    if dataset_name.startswith('mnist'):
        train_dataset = torchvision.datasets.MNIST(
            root='toy_data/', train=True, download=True, transform=torchvision.transforms.ToTensor())
        test_dataset = torchvision.datasets.MNIST(
            root='toy_data/', train=False, download=True, transform=torchvision.transforms.ToTensor())
    elif dataset_name.startswith('fmnist'):
        train_dataset = torchvision.datasets.FashionMNIST(
            root='toy_data/', train=True, download=True, transform=torchvision.transforms.ToTensor())
        test_dataset = torchvision.datasets.FashionMNIST(
            root='toy_data/', train=False, download=True, transform=torchvision.transforms.ToTensor())
    else:
        raise NotImplementedError
    
    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=1)
    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=1)

    train_cnn_acc, test_cnn_acc = train_cnn(
        train_set_loader, test_set_loader, train_dataset_loader, test_dataset_loader, device)
    train_mlp_acc, test_mlp_acc = train_mlp(
        train_set_loader, test_set_loader, train_dataset_loader, test_dataset_loader, device)
    train_log_rec_acc, test_log_rec_acc = train_log_reg(
        train_set_loader, test_set_loader, train_dataset_loader, test_dataset_loader, device)
    return train_cnn_acc, test_cnn_acc, train_mlp_acc, test_mlp_acc, train_log_rec_acc, test_log_rec_acc


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_dataset = FolderDataset(args.train_path, transform=lambda x: x / 255.)
    eval_dataset = FolderDataset(args.eval_path, transform=lambda x: x / 255.)
    train_queue = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=1)
    eval_queue = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=1)

    train_cnn_acc, test_cnn_acc, train_mlp_acc, test_mlp_acc, train_log_rec_acc, test_log_rec_acc = train_all_classifiers(train_queue, eval_queue, args.dataset, device, args.batch_size)
    print('Log reg train/test acc: %.4f %.4f' % (train_log_rec_acc, test_log_rec_acc))
    print('MLP train/testacc: %.4f %.4f' % (train_mlp_acc, test_mlp_acc))
    print('CNN train/test acc: %.4f %.4f' % (train_cnn_acc, test_cnn_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['mnist28', 'fmnist28'], required=True)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--eval_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    set_seeds(0, 0)

    main(args)
