import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_fn(main_folder_path):

    subfolders = [f.path for f in os.scandir(main_folder_path) if f.is_dir()]

    for folder in subfolders:
        activity = os.path.basename(folder)

        csv_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.csv')])

        if True:
            files = csv_files[:2]

            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            fig.suptitle(f'{activity}', fontsize=16)

            for i, filename in enumerate(files):
                file_path = os.path.join(folder, filename)
                df = pd.read_csv(file_path)

                # Plot
                df.plot(ax=axes[i], grid=True)
                axes[i].set_title(filename, fontsize=12)
                axes[i].set_xlabel("Sample Index")
                axes[i].set_ylabel("Accelerometer Value")
                axes[i].set_ylim(top=5) # Set max height to 5
                axes[i].legend(loc='lower right')

            plt.show()