import pandas as pd
from tensorflow.keras.models import load_model

class ChurnPipeline:
    def __init__(self):
        # Load trained model
        self.model = load_model("churn_model_best.h5")

        # Load feature structure
        df = pd.read_csv("churn_preprocessed.csv")
        self.columns = df.drop("Exited", axis=1).columns

    def predict(self, input_dict):
        df = pd.DataFrame([input_dict])

        # Ensure correct column order
        df = df[self.columns]

        prob = self.model.predict(df)[0][0]

        if prob > 0.5:
            return "❌ Customer WILL CHURN"
        else:
            return "✅ Customer will NOT churn"
