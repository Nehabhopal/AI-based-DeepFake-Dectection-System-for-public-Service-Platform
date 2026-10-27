# Model training guide

## Expected folders
`dataset/train/REAL`, `dataset/train/DEEPFAKE`, `dataset/val/REAL`, `dataset/val/DEEPFAKE`, `dataset/test/REAL`, and `dataset/test/DEEPFAKE`.

## Data quality
Use a documented, licensed dataset. Ensure labels are correct. Keep frames from the same source video and identities in a single split where possible. Avoid near-duplicate leakage. Keep test data untouched until final evaluation.

## Train
```powershell
python ml/train.py --epochs 8 --batch-size 16 --lr 0.0001
```
The default ResNet18 starts from torchvision's randomly initialized architecture (`weights=None`) so training does not silently depend on downloading pretrained weights. For stronger results, use a documented, legally permitted pretrained initialization if you deliberately change the implementation and record it.

The script uses seeded Python/NumPy/PyTorch RNGs, detects CUDA automatically, reports split counts, saves the best validation-loss checkpoint, training history, and model metadata. It fails clearly if required folders or either class are missing.

## Evaluate
```powershell
python ml/evaluate.py
```
Evaluation uses the test split and writes accuracy, per-class precision/recall/F1, macro-F1, confusion matrix and ROC-AUC when calculable. Metrics are not generated until the script runs successfully on real data.

## Interpretation
A high score on one dataset does not establish performance on unseen generators, compression, lighting, identities, or real-world images. Report dataset composition, split method, class balance, and uncertainty in any project presentation.
