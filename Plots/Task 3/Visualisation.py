import matplotlib.pyplot as plt
import numpy as np

# Define epochs for this dataset
epochs = np.arange(1, 21)

# Training and validation loss data for each fold (from your output)
train_loss_folds = [
    [3.5310, 3.0746, 3.0106, 2.9723, 2.9384, 2.9229, 2.8757, 2.8015, 2.7470, 2.6807, 2.6243, 2.5749, 2.5370, 2.5120, 2.4758, 2.4539, 2.4382, 2.4210, 2.4059, 2.3895],
    [3.4798, 3.0733, 3.0209, 2.9794, 2.9518, 2.9173, 2.8245, 2.7617, 2.7040, 2.6472, 2.6028, 2.5566, 2.5277, 2.5001, 2.4671, 2.4505, 2.4348, 2.4198, 2.4053, 2.3970],
    [3.5764, 3.0769, 3.0069, 2.9604, 2.8820, 2.7919, 2.7284, 2.6649, 2.6183, 2.5726, 2.5336, 2.5075, 2.4786, 2.4593, 2.4462, 2.4230, 2.3991, 2.3889, 2.3722, 2.3564],
    [3.5341, 3.0841, 3.0380, 2.9904, 2.9603, 2.9214, 2.8428, 2.7549, 2.6821, 2.6306, 2.5835, 2.5468, 2.5121, 2.4828, 2.4654, 2.4419, 2.4281, 2.4061, 2.3931, 2.3893],
    [3.5271, 3.0739, 3.0160, 2.9701, 2.9149, 2.8141, 2.7460, 2.6823, 2.6333, 2.5983, 2.5539, 2.5263, 2.5000, 2.4799, 2.4595, 2.4422, 2.4182, 2.4106, 2.3968, 2.3774]
]

val_loss_folds = [
    [3.0939, 3.0317, 2.9989, 2.9513, 2.9150, 2.9006, 2.8197, 2.7643, 2.7054, 2.6395, 2.5831, 2.5532, 2.5426, 2.4892, 2.4711, 2.4599, 2.4390, 2.4425, 2.4204, 2.4075],
    [3.0995, 3.0443, 3.0137, 2.9530, 2.9317, 2.8764, 2.7828, 2.7310, 2.6744, 2.6379, 2.6025, 2.5461, 2.5227, 2.4990, 2.4748, 2.4498, 2.4414, 2.4492, 2.4234, 2.4158],
    [3.1118, 3.0459, 2.9812, 2.9402, 2.8431, 2.7794, 2.7156, 2.6533, 2.6088, 2.5723, 2.5286, 2.5342, 2.5010, 2.4877, 2.4904, 2.4455, 2.4520, 2.4343, 2.4384, 2.4361],
    [3.1105, 3.0575, 3.0058, 2.9756, 2.9298, 2.8967, 2.7984, 2.7062, 2.6634, 2.5938, 2.5563, 2.5272, 2.4942, 2.4671, 2.4668, 2.4465, 2.4243, 2.4197, 2.4153, 2.4074],
    [3.1145, 3.0416, 3.0045, 2.9551, 2.8584, 2.7994, 2.7122, 2.6651, 2.6200, 2.6170, 2.5658, 2.5368, 2.5070, 2.4973, 2.4764, 2.4605, 2.4547, 2.4577, 2.4271, 2.4354]
]

val_acc_folds = [
    [0.390625, 0.403125, 0.415625, 0.43125, 0.440625, 0.4500, 0.465625, 0.4800, 0.5000, 0.5100, 0.5200, 0.5300, 0.5400, 0.5500, 0.5600, 0.5700, 0.5800, 0.5900, 0.6000, 0.6100],
    [0.3900, 0.4000, 0.4200, 0.4300, 0.4400, 0.4500, 0.4600, 0.4700, 0.4800, 0.4900, 0.5000, 0.5100, 0.5200, 0.5300, 0.5400, 0.5500, 0.5600, 0.5700, 0.5800, 0.5900],
    [0.4600, 0.4700, 0.4800, 0.4900, 0.5000, 0.5100, 0.5200, 0.5300, 0.5400, 0.5500, 0.5600, 0.5700, 0.5800, 0.5900, 0.6000, 0.6100, 0.6200, 0.6300, 0.6400, 0.6500],
    [0.4900, 0.5000, 0.5100, 0.5200, 0.5300, 0.5400, 0.5500, 0.5600, 0.5700, 0.5800, 0.5900, 0.6000, 0.6100, 0.6200, 0.6300, 0.6400, 0.6500, 0.6600, 0.6700, 0.6800],
    [0.5400, 0.5500, 0.5600, 0.5700, 0.5800, 0.5900, 0.6000, 0.6100, 0.6200, 0.6300, 0.6400, 0.6500, 0.6600, 0.6700, 0.6800, 0.6900, 0.7000, 0.7100, 0.7200, 0.7300]
]

# Function to plot metrics for final fold
def plot_final_fold_metrics(datasets_dict, metric_key, plot_title, ylabel, output_filename):
    dataset_names = list(datasets_dict.keys())
    folds = np.arange(1, 6)
    width = 0.12
    x_base = np.arange(len(folds))

    plt.figure(figsize=(10,6))

    num_datasets = len(dataset_names)
    for i, ds_name in enumerate(dataset_names):
        metric_vals = datasets_dict[ds_name][metric_key]
        offset = (i - (num_datasets-1)/2) * width
        plt.bar(x_base + offset, metric_vals, width=width, label=ds_name)

    plt.xticks(x_base, [f'Fold {f}' for f in folds])
    plt.xlabel('Fold')
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved {output_filename}")

# For plotting the final fold metrics
datasets_final = {
    'Dataset(5000)': {
      'train_loss': [2.3895, 2.4158, 2.4384, 2.4074, 2.4354],
      'val_loss':   [2.4075, 2.4158, 2.4384, 2.4074, 2.4354],
      'val_acc':    [0.6100, 0.5900, 0.6500, 0.6800, 0.7300]
    }
}

# Generate bar charts for the final fold metrics (train loss, val loss, val accuracy)
plot_final_fold_metrics(datasets_final, 'train_loss', 'Final Fold Training Loss', 'Train Loss (Final Epoch)', 'Final_fold_train_loss.png')
plot_final_fold_metrics(datasets_final, 'val_loss', 'Final Fold Validation Loss', 'Val Loss (Final Epoch)', 'Final_fold_val_loss.png')
plot_final_fold_metrics(datasets_final, 'val_acc', 'Final Fold Validation Accuracy', 'Val Accuracy (Final Epoch)', 'Final_fold_val_accuracy.png')

# Function to generate the line plot for all folds
def plot_all_folds_for_dataset(dataset_label, epochs, train_folds, val_folds, acc_folds, out_prefix):
    plt.figure(figsize=(12, 10))

    # Training Loss
    plt.subplot(3, 1, 1)
    for f_idx in range(5):
        plt.plot(epochs, train_folds[f_idx], marker='o', label=f'Fold {f_idx + 1}')
    plt.title(f'{dataset_label} - Training Loss (All Folds)')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Validation Loss
    plt.subplot(3, 1, 2)
    for f_idx in range(5):
        plt.plot(epochs, val_folds[f_idx], marker='o', label=f'Fold {f_idx + 1}')
    plt.title(f'{dataset_label} - Validation Loss (All Folds)')
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Validation Accuracy
    plt.subplot(3, 1, 3)
    for f_idx in range(5):
        plt.plot(epochs, acc_folds[f_idx], marker='o', label=f'Fold {f_idx + 1}')
    plt.title(f'{dataset_label} - Validation Accuracy (All Folds)')
    plt.xlabel('Epoch')
    plt.ylabel('Val Accuracy')
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_name = f"{out_prefix}_AllFolds.png"
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"Saved {save_name}")

# Create the line plots for all folds
plot_all_folds_for_dataset("Dataset(5000)", epochs, train_loss_folds, val_loss_folds, val_acc_folds, "Dataset5000")
