from pathlib import Path
import subprocess
from Utils.Settings import path_to_trodes_export
import numpy as np
from Utils.Settings import max_ISI_gap_recording
import torch
import matplotlib.pyplot as plt
from Utils.TrodesToPython.readTrodesExtractedDataFile3 import readTrodesExtractedDataFile
import os
from scipy.io.matlab import loadmat
from datetime import datetime,  timedelta
import pandas as pd

def check_single_rec_file(directory):

    # Initialize a counter for .rec files
    rec_file_count = 0
    rec_file_name = ""

    # Iterate through all items in the directory
    for file_path in directory.iterdir():
        # Check if the item is a file and ends with .rec
        if file_path.is_file() and file_path.name.endswith('.rec'):
            rec_file_count += 1
            rec_file_name = file_path.name
            if rec_file_count > 1:
                # Stop the search early if more than one .rec file is found
                return f"More than one .rec file found"

    # Check the final count of .rec files
    if rec_file_count == 1:
        print(f"Exactly one .rec file found: {rec_file_name}")
        return Path(directory, rec_file_name), rec_file_name
    elif rec_file_count == 0:
        return "No .rec files found."
    else:
        return "Error in counting .rec files."


def check_timestamp_gaps(raw_dat):

    # Calculate differences between consecutive timestamps
    intervals = np.diff(raw_dat.get_times())

    # Find where the intervals exceed the threshold
    gaps = intervals > max_ISI_gap_recording

    if any(gaps):
        print("Gaps detected at indices:", np.where(gaps)[0])
        return gaps
    else:
        print("No gaps detected.")

def get_mouse_name(directory):
    # Define the path

    # Find the parent of "ephys" using the parts of the path
    if "ephys" in directory.parts:
        ephys_index = directory.parts.index("ephys")
        mouse_name = directory.parts[ephys_index - 1] if ephys_index > 0 else None
    else:
        mouse_name = None

    return mouse_name

def get_recording_day(directory):
    filename = directory.stem  # This gets '20231213_180557'

    # Extract the date part from the filename
    date = filename.split('_')[0]  # Split on underscore and take the first part

    return date
def check_for_dio_folder(directory):


    # Iterate through all items in the directory
    for file_path in directory.iterdir():
        # Check if the item is a directory and ends with .DIO
        if file_path.is_dir() and file_path.name.endswith('.DIO'):
            return f"Folder with '.DIO' found: {file_path.name}"

    return "No folder with '.DIO' found."

def has_dio_folder(directory):
    # Iterate through all items in the directory
    for file_path in directory.iterdir():
        # Check if the item is a directory and ends with .DIO
        if file_path.is_dir() and file_path.name.endswith('.DIO'):
            print( f"'.DIO' folder alread available: {file_path.name}")
            return True
    print("No folder with '.DIO' found.")
    return False


def extract_DIO(path_recording_folder, path_recording):
    if not has_dio_folder(path_recording_folder):
        print("Extract DIO")
        command = f"{path_to_trodes_export} -rec {path_recording} -dio"
        # Run the command
        try:
            subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, text=True)
            print("Command executed successfully")
        except subprocess.CalledProcessError:
            print("An error occurred while executing the command.")

def has_analogio_folder(directory):
    # Iterate through all items in the directory
    for file_path in directory.iterdir():
        # Check if the item is a directory and ends with .DIO
        if file_path.is_dir() and file_path.name.endswith('.analog'):
            print( f"'.analog' folder alread available: {file_path.name}")
            return True
    print("No folder with '.analog' found.")
    return False

def has_time_folder(directory):
    # Iterate through all items in the directory
    for file_path in directory.iterdir():
        # Check if the item is a directory and ends with .DIO
        if file_path.is_dir() and file_path.name.endswith('.time'):
            print( f"'.time' folder alread available: {file_path.name}")
            return True
    print("No folder with '.time' found.")
    return False

def extract_analogIO(path_recording_folder, path_recording):
    if not has_analogio_folder(path_recording_folder):
        print("Extract analogIO")
        command = f"{path_to_trodes_export} -rec {path_recording} -analogio"
        # Run the command
        try:
            subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, text=True)
            print("Command executed successfully")
        except subprocess.CalledProcessError:
            print("An error occurred while executing the command.")

def extract_time(path_recording_folder, path_recording):
    if not has_time_folder(path_recording_folder):
        print("Extract timestamps")
        command = f"{path_to_trodes_export} -rec {path_recording} -time"
        # Run the command
        try:
            subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, text=True)
            print("Command executed successfully")
        except subprocess.CalledProcessError:
            print("An error occurred while executing the command.")

def find_mat_files_with_same_day(base_path, target_date):
    base_directory = Path(base_path)
    mat_files = []

    # Iterate over all items in the base directory
    for item in base_directory.iterdir():
        if item.is_dir():  # Ensure the item is a directory
            try:
                recording_day = get_recording_day(item)
                if recording_day == target_date:
                    # List all .mat files in the directory
                    for mat_file in item.glob('*.mat'):
                        print(f".mat file found: {mat_file}")
                        mat_files.append(mat_file)
            except Exception as e:
                print(f"Error processing {item}: {e}")

    return mat_files

def check_gpu_availability():
    if torch.cuda.is_available():
        print(f"GPU available: n = {torch.cuda.device_count()}")
        #return torch.device("cuda")
    else:
        "GPU not available"


def find_min_distance_TTL(array1, array2):
    min_distances = np.zeros_like(array1, dtype=np.float32)
    for i, time in enumerate(array1):
        # Find the insertion point
        pos = np.searchsorted(array2, time)

        # Compare to nearest lower and higher neighbors in array2
        candidates = []
        if pos > 0:  # There is a lower neighbor
            candidates.append(abs(time - array2[pos - 1]))
        if pos < len(array2):  # There is a higher neighbor
            candidates.append(abs(time - array2[pos]))

        # Store the minimum distance to array2 for this element of array1
        min_distances[i] = min(candidates)

    # Find the index of the maximum of these minimum distances
    index_of_most_distant_pulse = np.argmax(min_distances)

    # The most distant pulse in array1
    most_distant_pulse = array1[index_of_most_distant_pulse]
    most_distant_difference = min_distances[index_of_most_distant_pulse]

    print(f"The most distant pulse in array1 is at index {index_of_most_distant_pulse} with time {most_distant_pulse}")
    print(f"This pulse has a minimum distance of {most_distant_difference} to the closest pulse in array2")
    plt.plot(min_distances)
    return min_distances

def select_DIO_channel(path_DIO_folder):
    DIO_with_data = []
    for file in os.listdir(path_DIO_folder):
        DIO_dict = readTrodesExtractedDataFile(Path(path_DIO_folder, file))
        if len(DIO_dict['data'])>1:
            print(f"{file} contains data")
            DIO_with_data.append(DIO_dict)
    print(f"{len(DIO_with_data)} DIO files with data")
    if len(DIO_with_data)==1:
        DIO_dict = DIO_with_data[0]
    else:
        print("Multiple DIO files!")
    return DIO_dict


def stitch_bpod_times(bpod_file, day, DIO_timestamps_1_zeroed):
    block_n = 0
    trial_start_times = []
    trial_stop_times = []
    stimulus_block = []
    stimulus_name = []

    for n, file in enumerate(bpod_file):
        print(file)
        bpod_data = loadmat(file, simplify_cells=True)['SessionData']
        date_bpod = datetime.strptime(bpod_data["Info"]["SessionDate"], '%d-%b-%Y')
        date_trodes = datetime.strptime(day, '%Y%m%d')
        assert date_bpod == date_trodes, "Bpod and recording software days do not match."
        print(
            f"Bpod session started at {bpod_data['Info']['SessionStartTime_UTC']}, duration: {bpod_data['TrialEndTimestamp'][-1] / 60} min, ended at: {(datetime.strptime(bpod_data['Info']['SessionStartTime_UTC'], '%H:%M:%S') + timedelta(minutes=bpod_data['TrialEndTimestamp'][-1] / 60)).strftime('%H:%M:%S')}")  # not used in calculations


        if n == 0:
            trial_start_times.extend(bpod_data['TrialStartTimestamp'])
            trial_stop_times.extend(bpod_data['TrialEndTimestamp'])
        else:
            trial_start_times.extend(bpod_data['TrialStartTimestamp'] - bpod_data['TrialStartTimestamp'][0] + np.diff(
                DIO_timestamps_1_zeroed).max() + prev_last_start)
            trial_stop_times.extend(bpod_data['TrialEndTimestamp'] - bpod_data['TrialEndTimestamp'][0] + np.diff(
                DIO_timestamps_1_zeroed).max() + prev_last_stop)

        stimulus_name.extend(
            [bpod_data["Info"]["SessionProtocolBranchURL"].split("/")[-1]] * len(bpod_data['TrialEndTimestamp']))
        stimulus_block.extend(np.repeat(block_n, len(bpod_data['TrialEndTimestamp'])))
        prev_start_time = bpod_data["Info"]["SessionStartTime_MATLAB"]
        prev_last_start = bpod_data['TrialStartTimestamp'][-1]
        prev_last_stop = bpod_data['TrialEndTimestamp'][-1]
        block_n += 1

    trials = pd.DataFrame(
        {"bpod_start_time": trial_start_times, "bpod_stop_time": trial_stop_times, "stimulus_block": stimulus_block,
         "stimulus_name": stimulus_name})
    return trials