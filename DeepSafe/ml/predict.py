from pathlib import Path
import argparse
from backend.services.model_service import PredictionService

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("image")
    args=parser.parse_args()
    service=PredictionService(ROOT/"models"/"deepfake_resnet18_best.pth",ROOT/"models"/"model_metadata.json",ROOT/"outputs"/"gradcam")
    result=service.predict(Path(args.image), "cli_prediction")
    print({k:v for k,v in result.items() if not k.endswith("_path")})
if __name__=="__main__": main()
