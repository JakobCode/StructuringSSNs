# Parts of this script are taken from: https://github.com/bfortuner/pytorch_tiramisu.
#
# The source repository is under MIT License.
# Authors from original repository: Brendan Fortuner
#
# For more details and references checkout the repository and the readme of our repository.
#
# Author of this edited script: Anonymous 

import os
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
from CamVid.utils import imgs as img_utils
from torch.autograd import Variable

def iou(y_pred, y_true):

    # y_true = torch.reshape(y_true, [y_true.shape[0], -1])

    if len(y_pred.shape) == 4:
        y_pred = torch.reshape(y_pred.argmax(-1), [y_true.shape[0], -1])
    score = []

    # ignore background class 0
    for i in range(11):
        intersection = torch.sum((y_true == y_pred) * (y_pred == i))
        score.append(
            (intersection)
            / (
                torch.sum(y_true == i)
                + torch.sum((y_true != 11) * (y_pred == i))
                - intersection
            )
        )

    return torch.mean(torch.tensor(score))


def save_weights(WEIGHTS_PATH, weights_name, model, epoch, loss, err, iou):
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_name + ".cpkt")
    torch.save(
        {
            "startEpoch": epoch,
            "loss": loss,
            "error": err,
            "iou": iou,
            "state_dict": model.state_dict(),
        },
        weights_fpath,
    )


def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath, map_location="cpu")
    startEpoch = weights["startEpoch"]
    model.load_state_dict(weights["state_dict"])
    print(
        "loaded weights (lastEpoch {}, loss {}, error {}, iou {})".format(
            startEpoch - 1, weights["loss"], weights["error"], weights["iou"]
        )
    )
    return startEpoch


def get_predictions(output_batch):
    bs, h, w, c = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(-1)
    indices = indices.view(bs, h, w)
    return indices


def error(preds, targets):
    assert preds.size() == targets.size()
    bs, h, w = preds.size()
    n_pixels = bs * h * w
    incorrect = preds.ne(targets).cpu().sum()
    err = incorrect / n_pixels
    return torch.round(err, decimals=5)


def train(model, trn_loader, optimizer, criterion, epoch):
    model.train()
    trn_loss = 0
    trn_error = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for idx, data in enumerate(trn_loader):
        print(idx)

        inputs = Variable(data[0].to(device))
        targets = Variable(data[1].to(device))

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(targets, output)
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()
        pred = get_predictions(model.last_mean)
        trn_error += error(pred, targets.data.cpu())

    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    return trn_loss, trn_error


def test(model, test_loader, criterion, epoch=1, mean_type="predictive"):
    model.eval()
    test_loss = 0
    test_error = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    p_list = []
    t_list = []

    idx = 0
    with torch.no_grad():
        for data, target in test_loader:
            idx += 1
            print(idx)

            data = Variable(data.to(device))
            target = Variable(target.to(device))
            t_list.append(target.detach().cpu())
            output = model(data)
            test_loss += criterion(target, output).item()

            if mean_type == "predictive":
                pred = get_predictions(model.last_mean)
            elif mean_type == "softmax_sample_mean":
                pred = get_predictions(
                    torch.mean(
                        torch.nn.Softmax(dim=-1)(
                            torch.permute(output, [0, 1, 3, 4, 2])
                        ),
                        dim=1,
                    )
                )
            elif mean_type == "pred_sample_mean":
                pred = get_predictions(
                    torch.mean(
                        torch.nn.functional.one_hot(
                            torch.argmax(
                                torch.permute(output, [0, 1, 3, 4, 2]), dim=-1
                            ),
                            11,
                        ).float(),
                        dim=1,
                    )
                )
            p_list.append(pred.detach().cpu())
            test_error += error(pred, target.data.cpu())

    p_list = torch.cat(p_list)
    t_list = torch.cat(t_list)

    iou_val = iou(p_list, t_list)

    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    return test_loss, test_error, iou_val


def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()


def predict(
        model,
        input_loader,
        n_batches=1,
        save_folder=".plots/",
        print_results=False):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for input, target in input_loader:
        data = Variable(input.to(device))
        label = Variable(target.to(device))
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input, target, pred])

    return predictions


def view_sample_predictions(
    model, loader, n=9999999, save_path=None, pkl_path="./", save_pkl=True
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_counter = 0

    for inputs, targets in loader:
        data = Variable(inputs.to(device))
        label = Variable(targets.to(device))
        output = model(data, num_samples=1)
        pred = get_predictions(model.last_mean)
        batch_size = inputs.size(0)
        for i in range(batch_size):
            print(
                "Save prediction {} out of {}".format(
                    sample_counter + 1, 233))
            sample_counter += 1

            if save_path is not None:
                img_utils.save_image(
                    inputs[i],
                    save_path=os.path.join(
                        save_path,
                        "{}".format(sample_counter).zfill(4) + "_01_image.png",
                    ),
                )
                img_utils.save_annotated(
                    targets[i],
                    save_path=os.path.join(
                        save_path,
                        "{}".format(sample_counter).zfill(4) + "_02_label.png",
                    ),
                )
                img_utils.save_annotated(
                    pred[i],
                    save_path=os.path.join(
                        save_path,
                        "{}".format(sample_counter).zfill(4) +
                        "_03_prediction.png",
                    ),
                )

            if save_pkl:
                # 1 x NUM_SAMPLES x NUM_CLASSES x HEIGHT x WIDTH
                mean = model.last_mean[i]
                cov_diag = model.last_cov_diag[i]
                cov_factor = model.last_cov_fac[i]

                # save pickel files with all contents
                save_dict = {
                    # generel information
                    "sample_id": sample_counter,
                    "data": "CamVid",
                    # data
                    "sample": inputs[i].detach().cpu().numpy(),
                    "dim": inputs[i].detach().cpu().shape,
                    "labels": targets[i].numpy(),
                    "annotations": 1,
                    "num_classes": 11,
                    # model setup
                    "rank": model.rank,
                    # "model_path": model_path,
                    # prediction
                    "mean": mean.detach().cpu().numpy(),
                    "cov_diag": cov_diag.detach().cpu().numpy(),
                    "cov_factor": cov_factor.detach().cpu().numpy(),
                }

                with open(
                    os.path.join(
                        pkl_path,
                        f"{sample_counter}".zfill(4) + "_" + f"_information_file.pkl",
                    ),
                    "wb",
                ) as f:
                    pkl.dump(save_dict, f)

            if sample_counter == n:
                break

        if sample_counter == n:
            break
