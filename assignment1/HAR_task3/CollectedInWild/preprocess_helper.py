import os
import pandas as pd
import numpy as np

def resample(df):

    # Timedelta index
    df.set_index(pd.to_timedelta(df['time'], unit='s'), inplace=True)
    df.drop(columns=['time'], inplace=True)

    # Resample
    resampled_df = df.resample('20ms').mean()
    resampled_df.interpolate(method='linear', inplace=True)
        
    return resampled_df


def process_folders(inputdir, outputdir, transform):

    # Create output folder
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        print(f"Created output folder: {outputdir}")

    for subdirpath, _, filenames in os.walk(inputdir):
        # Create output subfolder
        relativesubdir = os.path.relpath(subdirpath, inputdir)
        outputsubdir = os.path.join(outputdir, relativesubdir)

        if not os.path.exists(outputsubdir):
            os.makedirs(outputsubdir)

        for filename in filenames:
            if filename.endswith(".csv"):
                path_inputfile = os.path.join(subdirpath, filename)
                path_outputfile = os.path.join(outputsubdir, filename)

                print(f"Processing file: {path_inputfile}")
                df = pd.read_csv(path_inputfile)

                # Transform
                processed_df = transform(df)

                # Save
                processed_df.to_csv(path_outputfile, index=False)


def split_folders(inputdir, transformations):

    for outputdir, task in transformations.items():

        # Create
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
            print(f"Created output folder: {outputdir}")

        for subdirpath, _, filenames in os.walk(inputdir):
            relativesubdir = os.path.relpath(subdirpath, inputdir)
            outputsubdir = os.path.join(outputdir, relativesubdir)

            if not os.path.exists(outputsubdir):
                os.makedirs(outputsubdir)

            for filename in filenames:
                if filename.endswith(".csv"):
                    path_inputfile = os.path.join(subdirpath, filename)
                    path_outputfile = os.path.join(outputsubdir, filename)

                    print(f"Processing {path_inputfile} for {outputdir}")

                    df = pd.read_csv(path_inputfile)
                    required_cols = task['columns']
                        
                    if all(col in df.columns for col in required_cols):
                        new_df = df[required_cols].copy()
                        new_df.rename(columns=task['rename_map'], inplace=True)
                            
                        new_df.to_csv(path_outputfile, index=False)
                    else:
                        print(f"Missing one or more columns")


def flip_signs(df):
    df_flipped = df.copy()
    for col in df_flipped.columns:
        df_flipped[col] = df_flipped[col] * -1
    return df_flipped

def add_one_to_accx(df):
    df_new = df.copy()
    if 'accx' in df_new.columns:
        df_new['accx'] = df_new['accx'] + 1
    return df_new




# ======= DRIVERS =======
def resample_driver():
    inputdir = "CollectedInWild\\Test"
    outputdir = "CollectedInWild\\TestMA50"

    process_folders(inputdir, outputdir, resample)
    print("\nResampling complete.")



def split_driver():
    # --- Splitting Accelrometer and Gyro & remapping the axis ---
    split_input_directory = "CollectedInWild\\TestMA50"
    split_transformations = {
         "CollectedInWild\\TestMA50_acc": {
            "columns": ["az", "ax", "ay"],
            "rename_map": {"ax": "accz", "ay": "accx", "az": "accy"}
        },
        "CollectedInWild\\TestMA50_both": {
            "columns": ["az", "ax", "ay",  "wz", "wx", "wy"],
            "rename_map": {
                "ax": "accz", "ay": "accx", "az": "accy",
                "wx": "gyroz", "wy": "gyrox", "wz": "gyroy"
            }
        }
    }

    split_folders(split_input_directory, split_transformations)
    print("\nSplitting and remapping complete.")





def flip_driver():
    # --- Flipping Signs ---
    flip_transformations = {
        "CollectedInWild\\TestMA50_acc": "CollectedInWild\\TestMA50F_acc",
        "CollectedInWild\\TestMA50_both": "CollectedInWild\\TestMA50F_both"
    }
    
    for input_dir, output_dir in flip_transformations.items():
        print(f"\nFlipping signs for files in '{input_dir}' and saving to '{output_dir}'")
        process_folders(input_dir, output_dir, flip_signs)

    print("\nSign flipping complete.")


    



def range_driver():
    # --- Range transformations ---
    add_one_transformations = {
        "CollectedInWild\\TestMA50F_acc": "CollectedInWild\\TestMA50F_acc_N",
        "CollectedInWild\\TestMA50F_mix": "CollectedInWild\\TestMA50F_mix_N"
    }

    for input_dir, output_dir in add_one_transformations.items():
        print(f"\nAdding 1 to accx for files in '{input_dir}' and saving to '{output_dir}'")
        process_folders(input_dir, output_dir, add_one_to_accx)
            
    print("Range Transforms complete.")

