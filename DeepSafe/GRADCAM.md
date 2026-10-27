# Grad-CAM implementation

The implementation hooks the ResNet18 final convolutional block (`layer4`), computes gradients of the predicted class score with respect to feature maps, averages gradients spatially to obtain channel weights, combines weighted activations, applies ReLU, normalizes, and resizes the heatmap. It saves both the heatmap and a blended overlay.

The heatmap is only generated after successful inference with a real trained checkpoint. It must not be described as proof that a face is fake. Red/yellow areas indicate stronger Grad-CAM activation for the selected class relative to the generated map; blue indicates lower activation. These colors are not probabilities or verified forensic artifacts.
