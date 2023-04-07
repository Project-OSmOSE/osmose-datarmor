from OSmOSE import Spectrogram
import argparse
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python script to process a list of file and generate spectrograms.")
    required = parser.add_argument_group('required arguments')
    required.add_argument("--input-file-list", "-l", required=True, help="The file path of the list of audio files, generally a timestamp.csv file.")
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

    with open(args.input_file_list, "r") as f:
        lines = f.readlines()
        
    adjust = args.nb_adjust_files and args.nb_adjust_files > 0
    if adjust:
        files_to_process = random.sample(lines, min(args.nb_adjust_files, len(lines) -1))
    else:
        files_to_process = lines[args.batch_ind_min: args.batch_ind_max if args.batch_ind_max != -1 else len(lines)]

    for audio_file in files_to_process:
        dataset.process_file(audio_file.rstrip(), adjust=adjust, save_matrix=args.save_matrix)
