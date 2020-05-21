import torch
import torchvision.models as models
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from petacc_patches_dataset import PetaccSplitDataset
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train inception model")

    parser.add_argument("-f", "--folder", type=str, help='Dataset folder path', default=None, required=True)

    cmdline_args = parser.parse_args()

    path = cmdline_args.folder
    batch_size_train = 32
    batch_size_test = 20

    test_set = PetaccSplitDataset(f"{path}\\validation.csv", f"{path}\\validate\\")
    test_loader = DataLoader(test_set, batch_size=batch_size_test, shuffle=True, num_workers=0)

    train_set = PetaccSplitDataset(f"{path}\\training.csv", f"{path}\\train\\")
    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("loading model")
    inception = models.inception_v3(pretrained=True)
    print("model loaded")

    # Replace the last layer
    inception.fc = nn.Sequential(
        nn.Linear(2048, 1),
        nn.Sigmoid()
    )

    inception.to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.SGD(inception.parameters(), lr=0.1, momentum=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1, verbose=True, min_lr=0.001)

    print("starting training")

    for epoch in range(30):
        inception.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch['image'].to(device), batch['annotation'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _ = inception(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            n = 5

            if i % n == (n - 1):  # print every n mini-batches
                print('\n[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / n))
                running_loss = 0.0

        inception.eval()
        correct = 0
        total = 0
        for i, batch in enumerate(test_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch['image'].to(device), batch['annotation'].to(device)

            # forward + backward + optimize
            outputs = inception(inputs)
            predictions = torch.round(outputs)
            correct += (labels == predictions).sum().item()
            total += labels.size(0)
        acc = correct / total
        print(f"After epoch no {epoch} the accuracy of the model is {acc}")
        scheduler.step(acc)
        idd = "exp1"
        torch.save(inception, f"{path}{idd}{epoch}")
