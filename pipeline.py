# Run AS-OCT LSCD CET segmentaiton pipeline

import os
import argparse
import warnings
import torch.distributed as dist
from pathlib import Path
from preprocessing import preprocess
from mask_creation import create_masks
from train import trainer
from postprocessing import postprocess
from evaluation import evaluate

def main():

    warnings.filterwarnings("ignore")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Set paths
    data_path = str(Path.cwd()) + '/data/'
    datasheet_path = str(Path.cwd()) + '/datasheet.xlsx'
    result_save_path = str(Path.cwd()) + '/results/'

    # Config for model parameters and training
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='MSU_Net', help='U_Net/MSU_Net')
    parser.add_argument('--model_path', type=str, default=result_save_path)
    parser.add_argument('--train_result_path', type=str, default=result_save_path)
    parser.add_argument('--val_result_path', type=str, default=result_save_path)
    parser.add_argument('--test_result_path', type=str, default=result_save_path)
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs_test', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank passed from torchrun')
    config = parser.parse_args()
    
    # Processing
    preprocess(data_path, result_save_path, config.folds)

    # Mask creation
    create_masks(data_path, result_save_path, datasheet_path)

    # Training and evaluation
    trainer(result_save_path, config, config.folds, datasheet_path)

    # Postprocessing
    postprocess(data_path, result_save_path)

    # Evaluation
    evaluate(result_save_path, datasheet_path)

    if not dist.is_initialized() or dist.get_rank() == 0:
        print('Finished, results saved to:', result_save_path)

if __name__ == "__main__":
    main()
