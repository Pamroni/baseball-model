from .xgb.xgb_model import XGBFangraphsModel
def train():
    model = XGBFangraphsModel()
    print("Loaded model, beginning training...")
    model.train()
    print("Training completed, saving model...")
    model.save_model()
    print("Model saved successfully.")

if __name__ == "__main__":
    train()