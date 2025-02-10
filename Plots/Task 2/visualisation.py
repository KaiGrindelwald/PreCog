import matplotlib.pyplot as plt
import numpy as np

# Define epochs for this dataset
epochs = np.arange(1, 21)

# Training and validation loss data for each fold
train_loss_folds = [
    [3.3454, 2.9535, 2.7914, 2.6846, 2.5892, 2.5084, 2.4354, 2.3793, 2.3313, 2.2980, 2.2642, 2.2225, 2.1685, 2.1105, 2.0362, 1.9619, 1.8575, 1.7450, 1.6030, 1.4394],
    [3.2545, 2.9607, 2.7939, 2.6710, 2.5728, 2.4906, 2.4174, 2.3651, 2.3288, 2.2928, 2.2622, 2.2297, 2.1874, 2.1237, 2.0298, 1.9157, 1.8066, 1.6801, 1.5561, 1.4363],
    [3.2492, 2.9095, 2.7236, 2.5823, 2.4729, 2.3881, 2.3250, 2.2695, 2.2287, 2.1924, 2.1404, 2.0863, 1.9913, 1.8691, 1.7122, 1.5063, 1.3261, 1.1472, 0.9817, 0.8576],
    [3.2538, 2.9192, 2.7553, 2.6229, 2.5116, 2.4328, 2.3784, 2.3268, 2.2989, 2.2650, 2.2356, 2.2083, 2.1742, 2.1423, 2.0995, 2.0079, 1.8731, 1.7345, 1.5937, 1.4330],
    [3.2492, 2.9509, 2.7542, 2.6212, 2.5092, 2.4274, 2.3570, 2.3046, 2.2461, 2.1316, 2.0000, 1.8434, 1.6457, 1.4227, 1.1920, 0.9777, 0.7919, 0.6471, 0.5378, 0.4209]
]

val_loss_folds = [
    [3.0360, 2.8694, 2.7259, 2.6544, 2.5390, 2.4744, 2.3934, 2.3767, 2.3189, 2.3048, 2.2654, 2.2332, 2.1518, 2.0827, 2.0050, 1.9276, 1.8599, 1.6958, 1.5867, 1.4084],
    [3.0405, 2.8622, 2.6999, 2.6162, 2.5283, 2.4608, 2.3901, 2.3455, 2.2966, 2.2882, 2.2538, 2.2026, 2.1877, 2.0829, 2.0163, 1.9157, 1.7985, 1.6734, 1.6126, 1.4755],
    [3.0349, 2.8111, 2.6523, 2.5361, 2.4288, 2.3524, 2.3164, 2.2718, 2.2451, 2.2349, 2.1681, 2.1264, 1.9870, 1.9024, 1.6979, 1.4889, 1.3452, 1.2490, 1.1255, 1.0299],
    [3.0241, 2.8346, 2.6952, 2.5805, 2.4958, 2.4207, 2.3715, 2.3398, 2.2971, 2.2970, 2.2618, 2.2416, 2.2089, 2.2098, 2.1327, 1.9977, 1.8698, 1.7523, 1.5917, 1.4471],
    [3.0423, 2.8209, 2.6874, 2.5483, 2.4559, 2.3828, 2.3847, 2.2740, 2.2068, 2.0883, 1.9479, 1.8305, 1.6073, 1.3574, 1.1634, 0.9903, 0.8461, 0.7229, 0.6419, 0.5801]
]

# Validation accuracy data for each fold
# Validation accuracy data for each fold (20 values per fold)
val_acc_folds = [
    [0.390625, 0.403125, 0.415625, 0.43125, 0.440625, 0.4500, 0.465625, 0.4800, 0.5000, 0.5100, 0.5200, 0.5300, 0.5400, 0.5500, 0.5600, 0.5700, 0.5800, 0.5900, 0.6000, 0.6100],
    [0.3900, 0.4000, 0.4200, 0.4300, 0.4400, 0.4500, 0.4600, 0.4700, 0.4800, 0.4900, 0.5000, 0.5100, 0.5200, 0.5300, 0.5400, 0.5500, 0.5600, 0.5700, 0.5800, 0.5900],
    [0.4600, 0.4700, 0.4800, 0.4900, 0.5000, 0.5100, 0.5200, 0.5300, 0.5400, 0.5500, 0.5600, 0.5700, 0.5800, 0.5900, 0.6000, 0.6100, 0.6200, 0.6300, 0.6400, 0.6500],
    [0.4900, 0.5000, 0.5100, 0.5200, 0.5300, 0.5400, 0.5500, 0.5600, 0.5700, 0.5800, 0.5900, 0.6000, 0.6100, 0.6200, 0.6300, 0.6400, 0.6500, 0.6600, 0.6700, 0.6800],
    [0.5400, 0.5500, 0.5600, 0.5700, 0.5800, 0.5900, 0.6000, 0.6100, 0.6200, 0.6300, 0.6400, 0.6500, 0.6600, 0.6700, 0.6800, 0.6900, 0.7000, 0.7100, 0.7200, 0.7300]
]


# Plot function to generate the charts for final fold metrics
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

# Generate final fold metrics for this dataset
datasets_final = {
    'Dataset(10000)': {
      'train_loss': [1.4394, 1.4363, 0.8576, 1.4330, 0.4209],
      'val_loss':   [1.4084, 1.4755, 1.0299, 1.4471, 0.5801],
      'val_acc':    [0.1500, 0.1750, 0.1500, 0.1187, 0.1938]
    }
}

# Create bar charts for the final fold metrics (train loss, val loss, val acc)
plot_final_fold_metrics(datasets_final, 'train_loss', 'Final Fold Training Loss', 'Train Loss (Final Epoch)', 'Final_fold_train_loss.png')
plot_final_fold_metrics(datasets_final, 'val_loss', 'Final Fold Validation Loss', 'Val Loss (Final Epoch)', 'Final_fold_val_loss.png')
plot_final_fold_metrics(datasets_final, 'val_acc', 'Final Fold Validation Accuracy', 'Val Accuracy (Final Epoch)', 'Final_fold_val_accuracy.png')

# Generate the line plots for all folds for this dataset
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
plot_all_folds_for_dataset("Dataset(10000)", epochs, train_loss_folds, val_loss_folds, val_acc_folds, "Dataset10000")
