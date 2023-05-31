import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
import os
from bottleneck import Vit_neck
from unet import Attention_UNet
from read_data import ReadData, color_params, color_transform_plot
from color_space_convert import *
from utils import *

class TrainAttentionUnet():
    def __init__(self, dataroot, valid_dataroot, image_size, color_space, net_dimension) -> None:

        # Local where data is for train
        self.dataroot = dataroot
        # Local where the validation data is 
        self.valid_dataroot = valid_dataroot
        # Size of the images
        self.image_size = image_size
        # Get the timestep to save the model and output images
        self.run_name = get_model_time()
        # Define the color space to use
        self.color_space = color_space
        actual_color_params = color_params(color_space)
        # The method t oconvert image from RGB to another color space
        self.color_transform = actual_color_params[0]
        # The normalization parameters for the chosen color space
        self.norm_params = actual_color_params[1]
        # The minimal number of channels in the network (the max is 16 times this number)
        self.net_dimension = net_dimension

    def read_datalaoder(self):
        """
        Get the data from the dataroot and return the dataloader
        """
        dataLoader = ReadData()
        dataloader = dataLoader.create_dataLoader(self.dataroot, self.image_size, self.color_transform, self.norm_params, batch_size=self.batch_size, shuffle=True)
        # If there is a validation dataset, return it
        if self.valid_dataroot:
            val_dataloader = dataLoader.create_dataLoader(self.val_dataset, self.image_size, self.color_transform, self.norm_params, batch_size=self.batch_size, shuffle=True)
            return dataloader, val_dataloader
        else:
            return dataloader
    
    def load_losses(self, loss):
        match loss:
            case "mse":
                criterion = nn.MSELoss()

        criterion = criterion.to(device)
        return criterion
    
    def train_epoch(self, model, dataloader, criterion, optimizer):
        """
        Method to train the model in one epoch and return the loss value
        """
        model.train()
        for img, img_gray, key_frame in dataloader:

            gt_img=img.to(device)
            sg_img=img_gray.to(device)

            ### Generate the color representation of the key frame of the video
            color = Vit_neck(key_frame).to(device)

            ### Generate the output image
            sc_img = model(sg_img, color)

            ### Meansure the difference between the generated image and the ground truth
            with torch.autocast(device_type=device):
                loss = criterion(sc_img, gt_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss, sc_img, gt_img
    
    def valid_epoch(self, model, device, dataloader, criterion):
        """
        Method to evaluate the model in the validation 
        dataset and return the loss value
        """
        model.eval()
        with torch.no_grad():
            for img, img_gray, key_frame in dataloader:

                gt_img=img.to(device)
                sg_img=img_gray.to(device)

                ### Generate the color representation of the key frame of the video
                color = Vit_neck(key_frame).to(device)

                ### Generate the output image
                sc_img = model(sg_img, color)

                ### Meansure the difference between the generated image and the ground truth
                with torch.autocast(device_type=device):
                    val_loss = criterion(sc_img, gt_img)

        return val_loss, sc_img
    
    def save_best_model(self, test_loss, best_loss, epoch, model, optimizer):
        """
        Method to save the best model and the best loss value in a txt file 
        """
        if test_loss < best_loss:

            ### Create the folder to save the model
            pos_path_save_models = os.path.join("models", self.run_name)
            os.makedirs(pos_path_save_models, exist_ok=True)

            best_loss = test_loss
            best_epoch = epoch

            ### Save the best model and optimizer
            checkpoint(model, os.path.join("models", self.run_name, "best_model.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", self.run_name, f"best_optimizer.pt"))

            return best_loss, best_epoch
    
    def train(self, epochs, lr, pretained_name=None):
        """
        Method to train the model and save the best model
        and the best loss value in a txt file 
        """
        ## Load datalaoders 
        train_loader, val_loader = self.read_datalaoder()

        ### Diffusion process
        model = Attention_UNet(img_size=self.image_size, net_dimension=self.net_dimension)

        best_loss = 999

        ## Read pretrained weights
        if pretained_name:
            # resume(diffusion_model, os.path.join("models", pretained_name, "ckpt.pt"))
            resume(model, os.path.join("models", pretained_name, "best_model.pt"))

        params_list = model.parameters()
        optimizer = optim.Adam(params_list, lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

        criterion = self.load_losses()

        logger = SummaryWriter(os.path.join("runs", self.run_name))

        ### Loop over the epochs
        epoch_pbar = tqdm(range(epochs), desc="Epochs", leave=True)
        for epoch in epoch_pbar:
            logging.info(f"Starting epoch {epoch}:")

            ## Train model
            loss, sc_img, gt_img = self.train_epoch(model, train_loader, criterion, optimizer)
            
            ## Evaluate model
            val_loss, val_sc_img = self.valid_epoch(model, val_loader, criterion)

            ### Update the logger
            logger.add_scalar("Loss", loss.item(), global_step=epoch + 1)

            epoch_pbar.set_postfix(MSE=loss.item(), MSE_val=val_loss.item(), lr=optimizer.param_groups[0]['lr'],  best_loss=best_loss)

            scheduler.step()

            if loss.item() < best_loss:
                test_loss = loss.item()
            elif val_loss.item() < best_loss:
                test_loss = val_loss.item()

            best_loss, best_epoch = self.save_best_model(self, test_loss, best_loss, epoch, model, optimizer)

            # elif epoch - best_epoch > early_stop_thresh:
            #     print(f"Early stopping at epoch {epoch}")
            #     break

            # resume(diffusion_model, os.path.join("unet_model", run_name, "best_model.pt"))
            # optimizer.load_state_dict(torch.load(os.path.join("unet_model", run_name, "best_optimizer.pt")))

            if epoch % 10 == 0:
                # Define the label size
                l = 5

                # Selecting the images to plot
                plot_gt = color_transform_plot(self.color_space, gt_img, l)
                plot_img = color_transform_plot(self.color_space, sc_img, l)
                val_plot_img = color_transform_plot(self.color_space, val_sc_img, l)
                
                ### Creating the Folders
                pos_path_save = os.path.join("models_resuts", self.run_name)
                os.makedirs(pos_path_save, exist_ok=True)

                ## Test if code is in a jupyternotebook, only print if yes
                if is_notebook():

                    ## Plot the ground truth
                    plot_images_2(plot_gt)
                    ## Plot the colorized version Sc
                    plot_images_2(plot_img)
                    ## Plot Validation
                    plot_images_2(val_plot_img)

                ## Save the Sc and ema img
                save_images_2(plot_img, os.path.join("unet_results", self.run_name, f"{epoch}.jpg"))
                save_images_2(val_plot_img, os.path.join("unet_results", self.run_name, f"{epoch}_val.jpg"))

                ### Save the models
                torch.save(model.state_dict(), os.path.join("unet_model", self.run_name, f"ckpt.pt"))
                torch.save(optimizer.state_dict(), os.path.join("unet_model", self.run_name, f"optimizer.pt"))

                ### Save the best loss info
                nome_arquivo = f"better_loss.txt"
                arquivo = open(os.path.join("unet_model", self.run_name, nome_arquivo), "w")
                arquivo.write(f"Epoch {best_epoch} - {best_loss}")
                arquivo.close()

        torch.cuda.empty_cache()
        logger.close()

if __name__ == "__main__":
    """
    Main function to train the model and save the best model and the best loss value in a txt file
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 224
    net_dimension = 18
    batch_size = 16
    epochs = 1000
    lr = 1e-4

    used_dataset = "mini_DAVIS"
    used_val_dataset = "mini_DAVIS_val"

    dataroot = f"\video_colorization\data\train{used_dataset}"
    valid_dataroot = f"\video_colorization\data\valid{used_val_dataset}"

    color_space = "Lab"
    train_model = TrainAttentionUnet(dataroot, valid_dataroot, image_size, color_space=color_space, net_dimension=net_dimension)
    train_model.train(epochs, lr)

    print("Done!")