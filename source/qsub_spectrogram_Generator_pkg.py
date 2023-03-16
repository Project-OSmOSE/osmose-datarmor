from OSmOSE import Spectrogram
import argparse
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python script to process a list of file and generate spectrograms.")
    required = parser.add_argument_group('required arguments')
    required.add_argument("--input-file-list", "-l", required=True, help="The file path of the list of audio files, generally a timestamp.csv file.")
    required.add_argument("--analysis-fs", "-fs", required=True, help="The analysis frequency.")
    required.add_argument("--dataset-path", "-p", required=True, help="The path to the dataset folder")
    parser.add_argument("--nb-adjust-files", "-a", type=int, help="The number of spectrograms to generated in order to adjust parameters. If a value higher than 0 is entered, the generation will switch to adjust mode. Default is 0.")
    parser.add_argument("--ind-min", "-min", type=int, default=0, help="The first file to consider. Default is 0.")
    parser.add_argument("--ind-max", "-max", type=int, default=-1, help="The last file to consider. -1 means consider all files from ind-min. Default is -1")

    args = parser.parse_args()

    dataset = Spectrogram(args.dataset_path, analysis_fs=args.analysis_fs)

    with open(args.input_file_list, "r") as f:
        lines = f.readlines()
        
    adjust = args.nb_adjust_files > 0
    if adjust:
        files_to_process = random.sample(lines, args.nb_adjust_files)
    else:
        files_to_process = lines[args.ind_min: args.ind_max if args.ind_max != -1 else len(lines)]

    for file in files_to_process:
        dataset.process_file(file, adjust=adjust)
