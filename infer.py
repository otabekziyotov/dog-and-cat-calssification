

import cv2, random, torch, torchmetrics, os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from train import TrainValidation
from transform import get_tfs

# # Define mean and std globally from the previously defined tfs object
# mean = val_tfs.transforms[2].mean
# std = val_tfs.transforms[2].std
# im_size = val_tfs.transforms[0].size[0] # Get the image size (96) from the first transform

class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Reverse the normalization applied to the image tensor.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class ModelInferenceVisualizer:
    def __init__(self, model, device, class_names=None, im_size=224,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], save_dir=None):

        self.denormalize = Denormalize(mean, std)
        self.model = model
        self.device = device
        self.class_names = class_names
        self.im_size = im_size
        # If save_dir is given -> save results to file
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        self.model.eval()  # Set model to evaluation mode

    def tensor_to_image(self, tensor):
        """
        Convert a normalized tensor to a denormalized image array.
        """
        tensor = self.denormalize(tensor)  # Denormalize the tensor
        tensor = tensor.permute(1, 2, 0)  # Convert from CxHxW to HxWxC
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def plot_value_array(self, logits, gt, class_names):
        """Plot the prediction probability array."""
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1)

        plt.grid(visible=True)
        plt.xticks(range(len(class_names)), class_names, rotation='vertical')
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        bars = plt.bar(range(len(class_names)), [p.item() for p in probs[0]], color="#777777")
        plt.ylim([0, 1])
        if pred_class.item() == gt:
            bars[pred_class].set_color('green')
        else:
            bars[pred_class].set_color('red')

    def generate_cam_visualization(self, image_tensor):
        """Generate GradCAM visualization."""
        # use_cuda is not needed: GradCAM detects the device from the model automatically
        # (CUDA if available -> CUDA, otherwise -> CPU)
        cam = GradCAMPlusPlus(model=self.model, target_layers=[self.model.features[-1].conv]) # resnet50 -> self.model.layer4[-1]
        grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0))[0, :]  # 2D, [4, 256], [256]
        return grayscale_cam

    def infer_and_visualize(self, test_dl, num_images=5, rows=2):
        """Perform inference and visualize predictions along with GradCAM."""
        preds, images, lbls, logitss = [], [], [], []
        accuracy, count = 0, 1

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(test_dl), desc="Inference"):
                # if idx == 2: break
                im, gt = TrainValidation.to_device(batch, device = self.device)
                # print(im.shape) # (bs, im_chs, im_h, im_w)
                # print(im[0].shape)
                logits = self.model(im)
                pred_class = torch.argmax(logits, dim=1)
                accuracy += (pred_class == gt).sum().item()
                images.append(im[0]) # (bs, im_chs, im_h, im_w) -> (im_chs, im_h, im_w)
                logitss.append(logits[0])
                preds.append(pred_class[0].item())
                lbls.append(gt[0].item())

        print(f"Accuracy of the model on the test data -> {(accuracy / len(test_dl.dataset)):.3f}")

        plt.figure(figsize=(20, 10))
        indices = [random.randint(0, len(images) - 1) for _ in range(num_images)]
        # for loop over 'num_images' images
        for idx, index in enumerate(indices):
            # Convert and denormalize image
            # print(images[index].shape)
            # rand = torch.rand(1, 3, 64, 64)
            # print(images[index].squeeze().shape)
            # print(f"rand.shape -> {rand.shape}")
            # print(f"rand.squeeze().shape -> {rand.squeeze().shape}")
            im = self.tensor_to_image(images[index]) # ()
            pred_idx = preds[index]
            gt_idx = lbls[index]

            # Display image
            plt.subplot(rows, 2 * num_images // rows, count)
            count += 1
            plt.imshow(im, cmap="gray")
            plt.axis("off")

            # GradCAM visualization
            grayscale_cam = self.generate_cam_visualization(images[index])
            visualization = show_cam_on_image(im / 255, grayscale_cam, image_weight=0.4, use_rgb=True)
            plt.imshow(cv2.resize(visualization, (self.im_size, self.im_size), interpolation=cv2.INTER_LINEAR), alpha=0.7, cmap='jet')
            plt.axis("off")

            # Prediction probability array
            logits = logitss[index]
            if logits.dim() == 1:  # If 1D, add a batch dimension
                logits = logits.unsqueeze(0)
            plt.subplot(rows, 2 * num_images // rows, count)
            count += 1
            self.plot_value_array(logits=logits, gt=gt_idx, class_names=self.class_names)

            # Title with GT and Prediction
            if self.class_names:
                gt_name = self.class_names[gt_idx]
                pred_name = self.class_names[pred_idx]
                color = "green" if gt_name == pred_name else "red"
                plt.title(f"GT -> {gt_name} ; PRED -> {pred_name}", color=color)

        # Save or show the inference images (GT/PRED + GradCAM)
        if self.save_dir:
            path = os.path.join(self.save_dir, "inference_results.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            print(f"Saqlandi -> {path}")

        # Plot confusion matrix
        plt.figure(figsize=(20, 10))
        cm = confusion_matrix(lbls, preds)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        if self.save_dir:
            path = os.path.join(self.save_dir, "confusion_matrix.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            print(f"Saqlandi -> {path}")
        else:
            plt.show()

# model = timm.create_model(model_name = model_name, pretrained = True, num_classes = len(classes)).to(device)
# model.load_state_dict(torch.load(f"{save_dir}/{save_prefix}_best_model.pth"))
# inference_visualizer = ModelInferenceVisualizer(
#     model=trainer.model,
#     device=device,
#     class_names=classes,  # classes is already a list of strings
#     im_size=im_size, # Use the actual image size from transforms
#     mean=mean,
#     std=std
# )

# inference_visualizer.infer_and_visualize(ts_dl, num_images = 20, rows = 4)