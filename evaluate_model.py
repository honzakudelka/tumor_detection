import torch
from torch.utils.data import DataLoader
from petacc_patches_dataset import PetaccSplitDataset
import argparse


def evaluate_model(model, data_loader, device):
    correct = 0
    total = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for batch in data_loader:
        predictions = torch.round(model(batch['image'].to(device))).cpu()

        truth = batch['annotation'] == 1

        tp += ((truth == predictions) & truth).sum().item()
        tn += ((truth == predictions) & ~truth).sum().item()
        fp += ((truth != predictions) & truth).sum().item()
        fn += ((truth != predictions) & ~truth).sum().item()

        correct += (truth == predictions).sum().item()
        total += truth.size(0)
        print("total: %d" % total)
        print("correct: %d" % correct)
    acc = correct / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("tp: %.3f" % tp)
    print("tn: %.3f" % tn)
    print("fp: %.3f" % fp)
    print("fn: %.3f" % fn)
    print("accuracy: %.3f" % acc)
    print("precision: %.3f" % precision)
    print("recall: %.3f" % recall)

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate inception model")

    parser.add_argument("-f", "--folder", type=str, help='Dataset folder path', default=None, required=True)
    parser.add_argument("-clf", "--classifier_path", type=str, help='Classifier path', default=None, required=True)

    cmdline_args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = cmdline_args.folder
    batch_size = 20
    model = torch.load(cmdline_args.classifier_path, map_location=torch.device(device))
    data_set = PetaccSplitDataset(f"{path}\\validation.csv", f"{path}\\validate\\")
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=0)

    evaluate_model(model, data_loader, device)
