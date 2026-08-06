import os
import glob
import torch
import torch.distributed as dist
import numpy as np
import random
import pandas as pd
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from solver import Solver
from dataloader import get_loader
from collections import defaultdict

def init_distributed_mode():
    """
    Initialize distributed training environment.
    """

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Single-GPU or CPU fallback
        rank, world_size, local_rank = 0, 1, 0

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    return device, rank, world_size, local_rank

def make_splits(DATASHEET_PATH, result_path, num_folds=5, seed=42):
    """
    Create stratified k-fold splits at the patient level based on severity.
    """
    # Load excel datasheet
    df = pd.read_excel(DATASHEET_PATH)
    file_names = df['Previous File Name'].tolist()
    file_names = [file[0:5]+file[26:28] for file in file_names] 
    severities = df['Severity'].tolist()

    # Group patients
    patient_to_severity = {}
    files_list = sorted(os.listdir(os.path.join(result_path, 'mask_overlays_outlines')))

    for file in files_list:
        num = int(n[-1])
        n = n[:-1] + str(num)
        severity = severities[file_names.index(n)]
        patient_id = file.split("_")[0]
        patient_to_severity[patient_id] = severity

    # Organize patients by severity
    class_patients = defaultdict(list)
    for pid, severity in patient_to_severity.items():
        class_patients[severity].append(pid)

    # Build stratified folds
    class_folds = {}
    np.random.seed(seed)
    for cls, patients in class_patients.items():
        shuffled = np.random.permutation(patients).tolist()
        class_folds[cls] = np.array_split(shuffled, num_folds)

    # Lookup patch files by patient
    images_dir = result_path + '/patches/images/'
    masks_dir = result_path + '/patches/masks/'
    all_image_files = os.listdir(images_dir)
    all_mask_files = os.listdir(masks_dir)

    patient_image_lookup = defaultdict(list)
    patient_mask_lookup = defaultdict(list)

    for fname in all_image_files:
        pid = fname.split("_")[0]
        patient_image_lookup[pid].append(fname)

    for fname in all_mask_files:
        pid = fname.split("_")[0]
        patient_mask_lookup[pid].append(fname)

    # Build final splits
    splits = []
    for fold_idx in range(num_folds):
        x_train, y_train = [], []
        x_val, y_val = [], []
        x_test, y_test = [], []

        train_classes = defaultdict(int)
        val_classes = defaultdict(int)
        test_classes = defaultdict(int)

        for cls, folds in class_folds.items():
            # Select test patients
            test_patients = folds[fold_idx].tolist()

            # Remaining patients for train+val
            remaining = [p for i, f in enumerate(folds) if i != fold_idx for p in f]

            # Validation: 10% of class total
            val_size = math.ceil(0.1 * len(class_patients[cls]))
            np.random.seed(seed + fold_idx)
            val_patients = np.random.choice(remaining, size=val_size, replace=False).tolist()
            train_patients = [p for p in remaining if p not in val_patients]

            # Add patients’ patches to splits
            for pid in train_patients:
                x_train += [os.path.join(images_dir, f) for f in patient_image_lookup[pid]]
                y_train += [os.path.join(masks_dir, f) for f in patient_mask_lookup[pid]]
                train_classes[cls] += 1

            for pid in val_patients:
                x_val += [os.path.join(images_dir, f) for f in patient_image_lookup[pid]]
                y_val += [os.path.join(masks_dir, f) for f in patient_mask_lookup[pid]]
                val_classes[cls] += 1

            for pid in test_patients:
                x_test += [os.path.join(images_dir, f) for f in patient_image_lookup[pid]]
                y_test += [os.path.join(masks_dir, f) for f in patient_mask_lookup[pid]]
                test_classes[cls] += 1

        splits.append({
            "fold": fold_idx + 1,
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
            "train_classes": dict(train_classes),
            "val_classes": dict(val_classes),
            "test_classes": dict(test_classes),
        })

    # Sanity check
    print("\nSanity Check (Patient-level splits):")
    for split in splits:
        counts = {k: len(split[k]) for k in ["x_train", "x_val", "x_test"]}
        print(f"\nFold {split['fold']}: Train={counts['x_train']} | Val={counts['x_val']} | Test={counts['x_test']}")
        print(f"  Train patients per class: {split['train_classes']}")
        print(f"  Val patients per class:   {split['val_classes']}")
        print(f"  Test patients per class:  {split['test_classes']}")

    # Ensure disjointness of test patients across folds
    all_test_patients = []
    for split in splits:
        test_patients = set()
        for f in split["x_test"]:
            test_patients.add(os.path.basename(f).split("_")[0])
        all_test_patients.append(test_patients)

    for i in range(len(all_test_patients)):
        for j in range(i + 1, len(all_test_patients)):
            assert all_test_patients[i].isdisjoint(all_test_patients[j]), f"Overlap between fold {i+1} and {j+1} test patients!"

    return splits

def set_seed(seed: int):
    """
    Set random number generator seeds for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def read_image_file(str):
    files_image = glob.glob(str + '/*.png')
    return files_image

def read_label_file(str):
    files_label = glob.glob(str + '/*.png')
    return files_label

def trainer(result_path, config, num_folds, datasheet_path):
    """
    Train+evaluate the model using Distributed Data Parallel (DDP) across multiple GPUs.
    """

    # Set random seeds for reproducibility
    set_seed(42)

    # Initialize distributed mode
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(
            backend="nccl", init_method="env://",
            world_size=world_size, rank=rank
        )
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0  # single GPU fallback

    splits = make_splits(datasheet_path, result_path, num_folds=num_folds)

    # Run for all folds
    for fold in range(num_folds):
        torch.backends.cudnn.benchmark = True

        config.model_path = os.path.join(result_path, f'UNet/UNet{fold+1}/')
        config.train_result_path = config.model_path
        config.val_result_path = config.model_path
        config.test_result_path = config.model_path

        if rank == 0:
            print(f"\n=== Fold {fold+1}/{num_folds} ===")
            print("Training with config:", config)

        fold_files = splits[fold]
        x_train, y_train = fold_files["x_train"], fold_files["y_train"]
        x_valid, y_valid = fold_files["x_val"], fold_files["y_val"]
        x_test, y_test   = fold_files["x_test"], fold_files["y_test"]

        total = len(x_train) + len(x_valid) + len(x_test)
        if rank == 0:
            print(
                f"Train: {len(x_train)/total:.2%}, "
                f"Valid: {len(x_valid)/total:.2%}, "
                f"Test: {len(x_test)/total:.2%}"
            )

        # Perform data loading
        train_loader, train_sampler = get_loader(
            imList=x_train, labelList=y_train, batch_size=config.batch_size,
            num_workers=config.num_workers, mode='train', drop_last=True,
            distributed=True
        )

        valid_loader, _ = get_loader(
            imList=x_valid, labelList=y_valid, batch_size=config.batch_size,
            num_workers=config.num_workers, mode='val', drop_last=False,
            distributed=False
        )

        test_loader, _ = get_loader(
            imList=x_test, labelList=y_test, batch_size=config.batch_size,
            num_workers=config.num_workers, mode='test', drop_last=False,
            distributed=False
        )

        solver = Solver(
            config, train_loader, valid_loader, test_loader,
            local_rank=local_rank, train_sampler=train_sampler
        )

        # Training
        solver.train()

        # Testing
        solver.test()

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
