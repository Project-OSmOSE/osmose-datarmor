from OSmOSE import Spectrogram
import argparse
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python script to process a list of file and generate spectrograms.")
    required = parser.add_argument_group('required arguments')
    required.add_argument("--sr-analysis", "-s", required=True, help="The analysis frequency.")
    required.add_argument("--dataset-path", "-p", required=True, help="The path to the dataset folder")
    parser.add_argument("--nb-adjust-files", "-a", type=int, help="The number of spectrograms to generated in order to adjust parameters. If a value higher than 0 is entered, the generation will switch to adjust mode. Default is 0.")
    parser.add_argument("--batch-ind-min", "-min", type=int, default=0, help="The first file to consider. Default is 0.")
    parser.add_argument("--batch-ind-max", "-max", type=int, default=-1, help="The last file to consider. -1 means consider all files from batch-ind-min. Default is -1")
    parser.add_argument("--save-matrix", "-m", action="store_true", help="Whether to save the spectrogram matrices or not. Note that activating this parameter might increase greatly the volume of the project.")

    args = parser.parse_args()

    print("Parameters :", args)

    os.system("ln -sf /appli/sox/sox-14.4.2_gcc-7.2.0/bin/sox sox")
    dataset = Spectrogram(args.dataset_path, sr_analysis=args.sr_analysis)

    if not dataset.path.joinpath("processed","spectrogram","adjust_metadata.csv"):
        raise FileNotFoundError(f"The file adjust_metadata.csv has not been found in the processed/spectrogram folder. Consider using the initialize() or update_parameters() methods.")

    files = list(dataset.audio_path.glob("*.wav"))

    print(f"Found {len(files)} files in {dataset.audio_path}.")
        
    adjust = args.nb_adjust_files and args.nb_adjust_files > 0
    if adjust:
        files_to_process = random.sample(files, min(args.nb_adjust_files, len(files) -1))
    else:
        files_to_process = files[args.ind_min: args.ind_max if args.ind_max != -1 else len(files)]

    for i, audio_file in enumerate(files_to_process):
        dataset.process_file(audio_file, adjust=adjust, save_matrix=args.save_matrix, clean_adjust_folder=True if i == 0 else False)
