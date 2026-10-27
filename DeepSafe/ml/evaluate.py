import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from ml.dataset import load_split, count_classes
from ml.model import build_model

ROOT=Path(__file__).resolve().parents[1]
def main():
    checkpoint_path=ROOT/"models"/"deepfake_resnet18_best.pth"
    if not checkpoint_path.is_file():
        raise SystemExit("Trained model is not available. Train first with: python ml/train.py")
    ds = load_split(ROOT / "dataset_rvf10k", "test")
    loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(2).to(device)
    ckpt=torch.load(checkpoint_path,map_location=device,weights_only=True)
    model.load_state_dict(ckpt.get("model_state_dict",ckpt)); model.eval()
    ys=[]; preds=[]; probs=[]
    with torch.no_grad():
        for x,y in loader:
            logits=model(x.to(device)); p=torch.softmax(logits,dim=1).cpu().numpy()
            ys.extend(y.numpy().tolist()); preds.extend(p.argmax(1).tolist()); probs.extend(p[:,1].tolist())
    precision,recall,f1,_=precision_recall_fscore_support(ys,preds,labels=[0,1],zero_division=0)
    metrics={"test_samples":len(ys),"class_counts":count_classes(ds),"accuracy":float(accuracy_score(ys,preds)),
             "classes":{"REAL":{"precision":float(precision[0]),"recall":float(recall[0]),"f1":float(f1[0])},
                        "DEEPFAKE":{"precision":float(precision[1]),"recall":float(recall[1]),"f1":float(f1[1])}},
             "macro_precision":float(np.mean(precision)),"macro_recall":float(np.mean(recall)),
             "macro_f1":float(np.mean(f1)),"confusion_matrix":confusion_matrix(ys,preds,labels=[0,1]).tolist()}
    try: metrics["roc_auc"]=float(roc_auc_score(ys,probs))
    except ValueError: metrics["roc_auc"]=None
    out=ROOT/"outputs"/"evaluation"; out.mkdir(parents=True,exist_ok=True)
    (out/"test_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(); plt.imshow(metrics["confusion_matrix"]); plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted class (REAL=0, DEEPFAKE=1)"); plt.ylabel("True class (REAL=0, DEEPFAKE=1)")
    plt.colorbar(); plt.tight_layout(); plt.savefig(out/"confusion_matrix.png"); plt.close()
    metadata_path=ROOT/"models"/"model_metadata.json"
    metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["evaluation_metrics"]=metrics
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
