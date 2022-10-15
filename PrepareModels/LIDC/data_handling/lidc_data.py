# All scripts in the folder ./Data/LIDC/data are taken from the Apache-2.0 lisensed repository
# https://github.com/MiguelMonteiro/PHiSeg-code
#
# Publication:
#
# Baumgartner, Christian F., et al.
# "Phiseg: Capturing uncertainty in medical image segmentation."
# International Conference on Medical Image Computing and Computer-Assisted Intervention.
# Springer, Cham, 2019.
#
#
# Code Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import LIDC.data_handling.lidc_data_loader as lidc_data_loader
import numpy as np
from LIDC.data_handling.batch_provider import BatchProvider


class lidc_data:
    def __init__(self, annotator_range, data_root, preproc_folder):

        data = lidc_data_loader.load_and_maybe_process_data(
            input_file=data_root,
            preprocessing_folder=preproc_folder,
            force_overwrite=False,
        )

        self.data = data
        num_labels_per_subject = 4
        nlabels = 2

        # Extract the number of training and testing points
        indices = {}
        for tt in data:
            N = data[tt]["images"].shape[0]
            indices[tt] = np.arange(N)

        # Create the batch providers
        # augmentation_options = exp_config.augmentation_options
        augmentation_options = {
            "do_flip_lr": True,
            "do_flip_ud": True,
            "do_rotations": True,
            "do_scaleaug": True,
            "nlabels": nlabels,
        }
        self.train = BatchProvider(
            data["train"]["images"],
            data["train"]["labels"],
            indices["train"],
            add_dummy_dimension=True,
            do_augmentations=True,
            augmentation_options=augmentation_options,
            num_labels_per_subject=num_labels_per_subject,
            annotator_range=annotator_range,
        )
        self.validation = BatchProvider(
            data["val"]["images"],
            data["val"]["labels"],
            indices["val"],
            add_dummy_dimension=True,
            num_labels_per_subject=num_labels_per_subject,
            annotator_range=annotator_range,
        )
        self.test = BatchProvider(
            data["test"]["images"],
            data["test"]["labels"],
            indices["test"],
            add_dummy_dimension=True,
            num_labels_per_subject=num_labels_per_subject,
            annotator_range=annotator_range,
        )

        self.test.images = data["test"]["images"]
        self.test.labels = data["test"]["labels"]

        self.validation.images = data["val"]["images"]
        self.validation.labels = data["val"]["labels"]


if __name__ == "__main__":

    # If the program is called as main, perform some debugging operations
    from phiseg.experiments import phiseg_7_5_4annot as exp_config

    data = lidc_data(exp_config)

    print(data.validation.images.shape)

    print(data.data["val"]["images"].shape[0])
    print(data.data["test"]["images"].shape[0])
    print(data.data["train"]["images"].shape[0])
    print(
        data.data["train"]["images"].shape[0]
        + data.data["test"]["images"].shape[0]
        + data.data["val"]["images"].shape[0]
    )

    print("DEBUGGING OUTPUT")
    print("training")
    for ii in range(2):
        X_tr, Y_tr = data.train.next_batch(10)
        print(np.mean(X_tr))
        print(Y_tr.shape)
        print("--")

    print("test")
    for ii in range(2):
        X_te, Y_te = data.test.next_batch(10)
        print(np.mean(X_te))
        print(Y_te.shape)
        print("--")

    print("validation")
    for ii in range(2):
        X_va, Y_va = data.validation.next_batch(10)
        print(np.mean(X_va))
        print(Y_va.shape)
        print("--")
