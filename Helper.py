import json
import numpy as np
import matplotlib.pyplot as plt


for split in ["training.json", "validation.json", "test.json"]:
    data = json.load(open(split))
    lengths = [ len(item["text"].split()) for item in data ]
    print(f"{split}: avg tokens = {np.mean(lengths):.1f}")
    
    
# Epoch indices
epochs = [1, 2, 3, 4, 5]

# Training and validation accuracies by epoch
train_accuracies = [0.529875, 0.583, 0.615875, 0.647375, 0.6455]
val_accuracies   = [0.54875,  0.5925, 0.57625,  0.59,    0.6025]

plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, val_accuracies,   label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Learning Curve (FFNN, h=100)")
plt.legend()
plt.savefig("learning_curve.png")
print("Saved learning_curve.png")