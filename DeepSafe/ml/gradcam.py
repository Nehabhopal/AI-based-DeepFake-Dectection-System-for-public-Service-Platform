
from pathlib import Path

import cv2
import numpy as np
import torch


def generate_gradcam(
    model,
    input_tensor,
    original_pil,
    class_index,
    heatmap_path,
    overlay_path,
    device,
):
    """
    Generate a Grad-CAM heatmap and overlay for an image.

    Args:
        model: Trained PyTorch model (ResNet18 architecture).
        input_tensor: Preprocessed image tensor with shape (1, C, H, W).
        original_pil: Original image as a PIL Image.
        class_index: Index of the class to explain.
        heatmap_path: Output path for the heatmap.
        overlay_path: Output path for the overlay.
        device: PyTorch device, such as CPU or CUDA.
    """

    activations = {}
    hook = None

    # Target the final convolutional layer in ResNet18.
    target_layer = model.layer4[-1].conv2

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    try:
        # Register a hook to capture the feature maps.
        hook = target_layer.register_forward_hook(forward_hook)

        # Grad-CAM needs gradients even if the caller uses
        # torch.no_grad() or torch.inference_mode().
        with torch.inference_mode(False), torch.enable_grad():

            model.eval()
            model.zero_grad(set_to_none=True)

            # Make a normal tensor outside inference mode.
            tensor = (
                input_tensor.detach()
                .clone()
                .to(device)
                .requires_grad_(True)
            )

            # Forward pass.
            logits = model(tensor)

            if "value" not in activations:
                raise RuntimeError(
                    "Grad-CAM failed to capture feature maps."
                )

            feature_maps = activations["value"]

            # Validate the requested class.
            if logits.ndim != 2:
                raise ValueError(
                    f"Expected model output shape (batch, classes), "
                    f"got {tuple(logits.shape)}."
                )

            if not 0 <= class_index < logits.shape[1]:
                raise ValueError(
                    f"Invalid class index {class_index}; "
                    f"model has {logits.shape[1]} classes."
                )

            # Obtain gradients directly from the selected class score.
            gradients = torch.autograd.grad(
                outputs=logits[:, class_index].sum(),
                inputs=feature_maps,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

            # Detach tensors for heatmap computation.
            acts = feature_maps.detach()
            grads = gradients.detach()

            # Average gradients across the spatial dimensions.
            weights = grads.mean(
                dim=(2, 3),
                keepdim=True,
            )

            # Weighted sum of feature maps followed by ReLU.
            cam = torch.relu(
                (weights * acts).sum(dim=1)
            )[0].cpu().numpy()

        # Normalize CAM values to [0, 1].
        cam = np.nan_to_num(
            cam,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        cam -= cam.min()
        max_value = cam.max()

        if max_value > 0:
            cam /= max_value

        # Resize to the original image dimensions.
        width, height = original_pil.size
        cam = cv2.resize(
            cam,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        # Convert the heatmap to a color image.
        heat = np.uint8(np.clip(cam * 255, 0, 255))
        colored = cv2.applyColorMap(
            heat,
            cv2.COLORMAP_JET,
        )

        # Convert the original image to OpenCV BGR format.
        rgb = np.asarray(
            original_pil.convert("RGB")
        )
        original_bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )

        # Blend the heatmap with the original image.
        overlay = cv2.addWeighted(
            original_bgr,
            0.58,
            colored,
            0.42,
            0,
        )

        # Create output directories.
        heatmap_path = Path(heatmap_path)
        overlay_path = Path(overlay_path)

        heatmap_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        overlay_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save both output images.
        if not cv2.imwrite(str(heatmap_path), colored):
            raise OSError(
                f"Failed to save Grad-CAM heatmap: {heatmap_path}"
            )

        if not cv2.imwrite(str(overlay_path), overlay):
            raise OSError(
                f"Failed to save Grad-CAM overlay: {overlay_path}"
            )

    finally:
        # Always remove the hook, even if generation fails.
        if hook is not None:
            hook.remove()

        model.zero_grad(set_to_none=True)
