import os
import json
import torch
import torch.nn as nn

# Paths
DATA_DIR = "data/processed"
MODEL_PATH = "model.pt"
PREDICTIONS_PATH = "predictions.json"


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 13 * 13, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Load model
model = SimpleCNN()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

# Load test data
test_images, test_labels = torch.load(
    os.path.join(DATA_DIR, "test.pt"), weights_only=False
)
test_images = test_images / 255.0

# Run inference
with torch.no_grad():
    outputs = model(test_images)
    _, predicted = torch.max(outputs, 1)

predicted_list = predicted.tolist()
labels_list = test_labels.tolist()

correct = sum(p == l for p, l in zip(predicted_list, labels_list))
accuracy = correct / len(labels_list)

print(f"Prediction accuracy: {accuracy:.4f}")

# Save predictions
results = {
    "accuracy": accuracy,
    "num_samples": len(predicted_list),
    "predictions": predicted_list[:100],
    "ground_truth": labels_list[:100],
}

with open(PREDICTIONS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print(f"Predictions saved to {PREDICTIONS_PATH}")
