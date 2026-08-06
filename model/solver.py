import os
import torch
import matplotlib.pyplot as plt
import torch.distributed as dist
import cv2
import numpy as np
import wandb
from torch import optim
from tqdm import tqdm, trange
from model import MSU_Net
from loss import DiceLoss
from concurrent.futures import ThreadPoolExecutor
from torch.nn.parallel import DistributedDataParallel as DDP

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader, local_rank=0, train_sampler=None):
        self.local_rank = int(local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')

        # Data
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.train_sampler = train_sampler

        # Paths & settings
        self.model_path = config.model_path
        self.train_result_path = config.train_result_path
        self.val_result_path = config.val_result_path
        self.test_result_path = config.test_result_path
        self.num_epochs = config.num_epochs
        self.num_epochs_tst = config.num_epochs_test
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.model_type = config.model_type

        # Build model and optimizer
        self.build_model()

    def build_model(self, load_weights=False):
        """
        Build the model, optimizer, and loss function.
        """
        torch.cuda.set_device(self.local_rank)
        self.unet = MSU_Net(img_ch=3, output_ch=1).to(self.device)

		# Load weights first if testing only
        if load_weights:
            unet_path = os.path.join(self.model_path, "model.pth")
            if os.path.exists(unet_path):
                state_dict = torch.load(unet_path, map_location=self.device)
                if any(k.startswith("module.") for k in state_dict.keys()):
                    state_dict = {k.replace("module.", ""): v for k,v in state_dict.items()}
                self.unet.load_state_dict(state_dict, strict=False)

    	# Wrap in DDP
        self.unet = DDP(self.unet, device_ids=[self.local_rank], output_device=self.local_rank
                        , find_unused_parameters=False)

		# Loss and optimizer (only needed for training)
        self.criterion = DiceLoss().to(self.device)
        self.optimizer = optim.SGD(self.unet.parameters(), self.lr, momentum=0.9, weight_decay=1e-6)

    def reset_grad(self):
        self.optimizer.zero_grad()

    def train(self):
        """
        Train the model using Distributed Data Parallel (DDP) across multiple GPUs.
        """
        best_score = 0.0

        # Track losses
        train_total_losses = []
        val_total_losses = []

        # Track validation dice score
        val_dice_scores= []

        outer = trange(self.num_epochs, desc="Overall Epoch Progress") if dist.get_rank() == 0 else range(self.num_epochs)
        alpha = 0
        for epoch in outer:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self.unet.train()
            total_loss_epoch, seg_loss_epoch = 0.0, 0.0

            train_iter = (
                tqdm(self.train_loader, desc=f"[Epoch {epoch+1}/{self.num_epochs}] Training", 
                    leave=False, total=len(self.train_loader))
                    if dist.get_rank() == 0 else self.train_loader)

            # Training loop
            for images, GT, _ in train_iter:
                images = images.to(self.device, non_blocking=True)
                GT = GT.float().clamp(0, 1).to(self.device, non_blocking=True)

                # Generate predictios
                im_preds = self.unet(images)

                # Apply sigmoid to segmentation predictions
                SR = torch.sigmoid(im_preds)

                # Compute dice loss from segmentation predictions
                dice_loss = self.criterion(SR.view(SR.size(0), -1), GT.view(GT.size(0), -1))

                # Add Dice, BCE, and the agreement loss terms together
                loss =  dice_loss

                total_loss_epoch += loss.item()
                seg_loss_epoch += dice_loss.item()

                self.reset_grad()
                loss.backward()
                self.optimizer.step()

            # Average losses
            avg_total_loss = total_loss_epoch / len(self.train_loader)

            # Update loss
            train_total_losses.append(avg_total_loss)

            if dist.get_rank() == 0:
                print(f"[Epoch {epoch+1}/{self.num_epochs}] Train Total: {avg_total_loss:.4f}") 

            # Validation loop
            self.unet.eval()
            val_total = 0.0
            v_DC = 0.0 

            val_iter = (
                tqdm(self.valid_loader, desc=f"[Epoch {epoch+1}/{self.num_epochs}] Validation", 
                    leave=False, total=len(self.valid_loader))
                    if dist.get_rank() == 0 else self.valid_loader)

            with torch.no_grad():
                for images, GT, _  in val_iter:
                    images = images.to(self.device, non_blocking=True)
                    GT = GT.float().clamp(0, 1).to(self.device, non_blocking=True)

                    im_preds = self.unet(images)
                    SR = torch.sigmoid(im_preds)

                    # Compute dice loss from segmentation predictions
                    dice_loss = self.criterion(SR.view(SR.size(0), -1), GT.view(GT.size(0), -1))

                    # Add Dice, BCE, and the agreement loss terms together
                    total_loss = dice_loss

                    val_total += total_loss.item()

                    # Segmentation metrics
                    v_DC += get_DC(SR, GT)

            # Average validation metrics
            avg_val_total = val_total / len(self.valid_loader)
            avg_dice = v_DC / len(self.valid_loader)

            val_total_losses.append(avg_val_total)
            val_dice_scores.append(avg_dice)
            avg_score = (v_DC) / len(self.valid_loader)

            if dist.get_rank() == 0:
                print(f"[Epoch {epoch+1}/{self.num_epochs}] Val Total: {avg_val_total:.4f} | Dice: {avg_dice:.4f}")

            # Save best model
            if dist.get_rank() == 0 and avg_score > best_score:
                best_score = avg_score
                os.makedirs(self.model_path, exist_ok=True)
                torch.save(self.unet.state_dict(), os.path.join(self.model_path, "best_model.pth"))
                print(f"Saved best model at epoch {epoch+1}, Score: {best_score:.4f}")
                wandb.run.summary["best_val_score"] = best_score

            # Save final model at the last epoch
            if epoch+1 == self.num_epochs:
                best_score = avg_score
                os.makedirs(self.model_path, exist_ok=True)
                torch.save(self.unet.state_dict(), os.path.join(self.model_path, "final_model.pth"))
                print(f"Saved final model at epoch {epoch+1}, Score: {best_score:.4f}")

            # Plot loss and metrics after each epoch
            if dist.get_rank() == 0:
                os.makedirs(self.model_path, exist_ok=True)

                # Loss
                plt.figure()
                plt.plot(range(1, len(train_total_losses)+1), train_total_losses, label="Train Total Loss")
                plt.plot(range(1, len(val_total_losses)+1), val_total_losses, label="Val Total Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Total Loss")
                plt.legend()
                plt.savefig(os.path.join(self.model_path, "Total_Loss.png"))
                plt.close()

                # Dice score
                plt.figure()
                plt.plot(range(1, len(val_dice_scores)+1), val_dice_scores, label="Dice Score")
                plt.xlabel("Epoch")
                plt.ylabel("Score")
                plt.title("Dice Score")
                plt.legend()
                plt.savefig(os.path.join(self.model_path, "Scores.png"))
                plt.close()
        
        if dist.get_rank() == 0:
            wandb.finish()

    def test(self):
        """
        Test the model using Distributed Data Parallel (DDP) across multiple GPUs.
        """

        load_path = os.path.join(self.model_path, "best_model.pth")

        state_dict = torch.load(load_path, map_location=self.device)

        # Handle DDP prefix
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.unet.module.load_state_dict(state_dict, strict=False)
        self.unet.eval()

        os.makedirs(os.path.join(self.test_result_path, "predicted_masks"), exist_ok=True)

        test_iter = tqdm(self.test_loader, desc="Testing", total=len(self.test_loader))

        def save_mask(pred_mask, save_path):
            np.save(save_path, pred_mask)

        executor = ThreadPoolExecutor(max_workers=8)

        # Run inference on test set and save predicted masks
        with torch.inference_mode():
            for images, _, filenames in test_iter:
                images = images.to(self.device, non_blocking=True)
                im_preds = self.unet(images)
                SR = torch.sigmoid(im_preds)
                SR_binary = (SR > 0.5).float() * 255

                for b in range(images.size(0)):
                    filename = filenames[b]
                    pred_mask = SR_binary[b].cpu().squeeze(0).numpy().astype(np.uint8)

                    # Resize to 100x100 pixels
                    pred_mask_resized = cv2.resize(pred_mask, (100, 100), interpolation=cv2.INTER_NEAREST)

                    # Save the predicted mask as a .npy file
                    save_path = os.path.join(
                        self.test_result_path, "predicted_masks", filename[0:-4].replace(".png", ".npy"))

                    executor.submit(save_mask, pred_mask_resized, save_path)
