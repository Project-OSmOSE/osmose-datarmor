

import argparse

import os

import random

from typing import Tuple

from warnings import warn

import numpy as np

import soundfile as sf

def safe_read(

    file_path: str, *, nan: float = 0.0, posinf: any = None, neginf: any = None

) -> Tuple[np.ndarray, int]:

    """Open a file using Soundfile and clean up the data to be used safely

    Currently, only checks for `NaN`, `inf` and `-inf` presence. The default behavior is the same as `np.nan_to_num`:

    `NaNs` are transformed into 0.0, `inf` and `-inf` are transformed into the maximum and minimum values of their dtype.

    Parameters

    ----------

        file_path: `str`

            The path to the audio file to safely read.

        nan: `float`, optional, keyword_only

            The value that will replace `NaNs`. Default is 0.0

        posinf: `any`, optional, keyword_only

            The value that will replace `inf`. Default behavior is the maximum value of the data type.

        neginf: `any`, optional, keyword_only

            The value that will replace `-inf`. Default behavior is the minimum value of the data type.

    Returns

    -------

        audio_data: `NDArray`

            The cleaned audio data as a numpy array.

        sample_rate: `int`

            The sample rate of the data."""

    audio_data, sample_rate = sf.read(file_path)

    nan_nb = sum(np.isnan(audio_data))


    if nan_nb > 0:

        warn(

            f"{nan_nb} NaN detected in file {os.path.basename(file_path)}. They will be replaced with {nan}."

        )

    np.nan_to_num(audio_data, copy=False, nan=nan, posinf=posinf, neginf=neginf)


    return audio_data, sample_rate


def check_n_files(

        dataset_path,

        file_list: list,

        n: int,

        *,

        threshold_percent: float = 0.1,

        auto_normalization: bool = False,

    ) -> bool:

        """Check n files at random for anomalies and may normalize them.

        Currently, check if the data for wav in PCM float format are between -1.0 and 1.0. If the number of files that

        fail the test is higher than the threshold (which is 10% of n by default, with an absolute minimum of 1), all the

        dataset will be normalized and written in another file.

        Parameters

        ----------

            file_list: `list`

                The list of files to be evaluated. It must be equal or longer than n.

            n: `int`

                The number of files to evaluate. To lower resource consumption, it is advised to check only a subset of the dataset.

                10 files taken at random should provide an acceptable idea of the whole dataset.

            threshold_percent: `float`, optional, keyword-only

                The maximum acceptable percentage of evaluated files that can contain anomalies. Understands fraction and whole numbers. Default is 0.1, or 10%

            auto_normalization: `bool`, optional, keyword_only

                Whether the normalization should proceed automatically or not if the threshold is reached. As a safeguard, the default is False.

        Returns

        -------

            normalized: `bool`

                Indicates whether or not the dataset has been normalized.

        """

        if threshold_percent > 1:

            threshold_percent = threshold_percent / 100


        file_list = file_list.split(" ")

        if "float" in str(sf.info(file_list[0])):

            threshold = max(threshold_percent * n, 1)

            bad_files = []

            for audio_file in random.sample(file_list, n):

                data, sr = safe_read(audio_file)

                if not (np.max(data) < 1.0 and np.min(data) > -1.0):

                    print("bad file:", audio_file)

                    bad_files.append(audio_file)


                    if len(bad_files) > threshold:

                        print(

                            "The treshold has been exceeded, too many files unadequately recorded."

                        )

                        if not auto_normalization:

                            raise ValueError(

                                "You need to set auto_normalization to True to normalize your dataset automatically."

                            )

                        os.makedirs(os.path.join(

                                    dataset_path,

                                    "raw",

                                    "audio",

                                    "normalized_original"))

                        for audio_file in file_list:

                            data, sr = safe_read(audio_file)

                            data = (

                                (data - np.mean(data)) / np.std(data)

                            ) * 0.063  # = -24dB

                            data[data > 1] = 1

                            data[data < -1] = -1

                            sf.write(

                                os.path.join(

                                    dataset_path,

                                    "raw",

                                    "audio",

                                    "normalized_original",

                                    os.path.basename(audio_file),

                                ),

                                data=data,

                                samplerate=sr,

                            )

                            # TODO: lock in spectrum mode

                        print(

                            "All files have been normalized. Spectrograms created from them will be locked in spectrum mode."

                        )

                        return True

        print("nothing to normalize")

        return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--path")

    parser.add_argument("--input-files")

    parser.add_argument("-n")

    parser.add_argument("--threshold")


    args = parser.parse_args()

    check_n_files(args.path, args.input_files, int(args.n), threshold_percent=float(args.threshold), auto_normalization=True)

