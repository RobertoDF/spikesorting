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
from matplotlib.patches import Rectangle
import seaborn as sns
import matplotlib.patches as mpatches
import panel as pn
from Utils.Settings import channel_label_color_dict
import spikeinterface.widgets as sw
import re
from datetime import time, date
from IPython.display import display, HTML
import sys

# Function to print in color
def print_in_color(text, color):
    color_text = f"<span style='color:{color}'>{text}</span>"
    display(HTML(color_text))

def get_recording_time(path_recording_folder):
    # Convert the last part of the path (filename) to a string
    filename = path_recording_folder.name
    time_match = re.search(r'_(\d{2})(\d{2})(\d{2})', filename)

    hours, minutes, seconds = map(int, time_match.groups())  # Convert each group to integer

    # Create a datetime.time object
    extracted_time = time(hours, minutes, seconds)
    return extracted_time

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
        try:
            if "win" in sys.platform:
                subprocess.run(command, check=True, shell=False, text=True)
                print("Command executed successfully")
            else:
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

def find_mat_files_with_same_day(base_path, path_recording_folder, raw_rec):
    time_rec = get_recording_time(path_recording_folder)
    end_time_rec = (datetime.combine(date.today(),  time_rec ) + timedelta(seconds=raw_rec.get_total_duration())).time() # we need a date to add times
    target_date = get_recording_day(path_recording_folder)# day of rec
    start_target_time = get_recording_time(path_recording_folder)# time of rec

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
                        bpod_data = loadmat(mat_file, simplify_cells=True)['SessionData']
                        start_time_mat_file  = datetime.strptime(bpod_data['Info']['SessionStartTime_UTC'], '%H:%M:%S').time()
                        if (start_time_mat_file> start_target_time) and (start_time_mat_file< end_time_rec):
                            print("Bpod file starts within the Trodes recording")
                            mat_files.append((mat_file, start_time_mat_file))
                        else:
                            print_in_color("Bpod file starts outside the Trodes recording!", "red")
            except Exception as e:
                print(f"Error processing {item}: {e}")

    mat_files.sort(key=lambda x: x[1])  # x[1] is the time part of the tuple
    return [file for file, _ in mat_files]  # Return only the file paths


def check_gpu_availability():
    if torch.cuda.is_available():
        print(f"GPU available: n = {torch.cuda.device_count()}")
        #return torch.device("cuda")
    else:
        "GPU not available"

def find_min_distance_TTL(DIO_samples_start_trial, start_times_bpod):
    print_in_color(f"len DIO:{len(DIO_samples_start_trial)}, len Bpod:{len(start_times_bpod)}", "red")

    array1 = (DIO_samples_start_trial - DIO_samples_start_trial[0]).copy()

    start_times_bpod_zeroed = start_times_bpod - start_times_bpod[0]
    array2 = start_times_bpod_zeroed

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

    fig, ax = plt.subplots()
    ax.plot(min_distances)
    ax.set_title("Distances between Bpod and Trodes trial start time" , fontsize=16, fontweight='bold')

    ax.text(0.5, 0.94, "There should be no abrupt spikes", transform=ax.transAxes, ha='center', fontsize=10, style='italic', color="red")
    ax.set_xlabel("Trial_n")
    ax.set_ylabel("Time (s)")
    sns.despine(ax=ax)
    return min_distances

def select_DIO_channel(path_DIO_folder):
    DIO_with_data = []
    for file in os.listdir(path_DIO_folder):
        DIO_dict = readTrodesExtractedDataFile(Path(path_DIO_folder, file))
        if len(DIO_dict['data'])>2:
            print(f"{file} contains data")
            DIO_with_data.append(DIO_dict)
    print(f"{len(DIO_with_data)} DIO files with data")
    assert len(DIO_with_data)==1, "Multiple DIO files!"
    DIO_dict = DIO_with_data[0]

    return DIO_dict


def stitch_bpod_times(bpod_file, day, DIO_timestamps_start_trial):
    assert len(bpod_file)<3, "More than 2 bpod files!"
    DIO_timestamps_start_trial_zeroed = DIO_timestamps_start_trial - DIO_timestamps_start_trial[0]
    block_n = 0
    trial_start_times = []
    trial_stop_times = []
    stimulus_block = []
    stimulus_name = []
    collect_gaps_between_blocks = []
    trials_data_dfs = []
    for n, file in enumerate(bpod_file):
        print(file)
        bpod_data = loadmat(file, simplify_cells=True)['SessionData']
        date_bpod = datetime.strptime(bpod_data["Info"]["SessionDate"], '%d-%b-%Y')
        date_trodes = datetime.strptime(day, '%Y%m%d')
        assert date_bpod == date_trodes, "Bpod and recording software days do not match."
        print(
            f"Bpod session started at {bpod_data['Info']['SessionStartTime_UTC']}, duration: {bpod_data['TrialEndTimestamp'][-1] / 60} min, ended at: {(datetime.strptime(bpod_data['Info']['SessionStartTime_UTC'], '%H:%M:%S') + timedelta(minutes=bpod_data['TrialEndTimestamp'][-1] / 60)).strftime('%H:%M:%S')}")  # not used in calculations
        print(f"number trials: {len(bpod_data['TrialStartTimestamp'])}")
        if n == 0:
            trial_start_times.extend(bpod_data['TrialStartTimestamp'])
            trial_stop_times.extend(bpod_data['TrialEndTimestamp'])
        else:
            trial_start_times.extend(bpod_data['TrialStartTimestamp'] - bpod_data['TrialStartTimestamp'][0] + np.diff(
                DIO_timestamps_start_trial_zeroed).max() + prev_last_start)
            trial_stop_times.extend(bpod_data['TrialEndTimestamp'] - bpod_data['TrialEndTimestamp'][0] + np.diff(
                DIO_timestamps_start_trial_zeroed).max() + prev_last_stop)
            collect_gaps_between_blocks.append([prev_last_start, np.diff(
                DIO_timestamps_start_trial_zeroed).max()])
        stimulus_name.extend(
            [bpod_data["Info"]["SessionProtocolBranchURL"].split("/")[-1]] * len(bpod_data['TrialEndTimestamp']))
        stimulus_block.extend(np.repeat(block_n, len(bpod_data['TrialEndTimestamp'])))
        prev_last_start = bpod_data['TrialStartTimestamp'][-1]
        prev_last_stop = bpod_data['TrialEndTimestamp'][-1]
        block_n += 1

        # extract trial data
        if "AuditoryTuning" in str(file):
            print("Extracting AuditoryTuning params")
            TrialData_dict = {key: value for key, value in bpod_data["Custom"].items() if
                              key in ['Frequency', "Volume"]}
            trial_data_df = pd.DataFrame(TrialData_dict)
        elif "DetectionConfidence" in str(file):
            print("Extracting DetectionConfidence params")
            TrialData_dict = {key: value for key, value in bpod_data["Custom"]["TrialData"].items() if
                              not isinstance(value, int)}

            TrialData_dict = {key: value[:bpod_data["nTrials"]] for key, value in TrialData_dict.items() if
                              len(value) == bpod_data["nTrials"] or len(value) == bpod_data["nTrials"] + 1}
            trial_data_df = pd.DataFrame(TrialData_dict)

        assert bpod_data["nTrials"] == len(trial_data_df)
        trials_data_dfs.append(trial_data_df)

    trials = pd.concat([pd.DataFrame(
        {"bpod_start_time": trial_start_times, "bpod_stop_time": trial_stop_times, "stimulus_block": stimulus_block,
         "stimulus_name": stimulus_name}),  pd.concat(trials_data_dfs, ignore_index=True)], axis=1)

    fig, ax = plt.subplots()
    ax.set_title("Check gap was correctly identified")
    trials.plot.scatter(x="bpod_start_time", y="stimulus_name", s=5, ax=ax)

    ylim = ax.get_ylim()

    if len(collect_gaps_between_blocks)>0:
        ax.text(0.5, 0.5, f"N={ len(collect_gaps_between_blocks)} gaps found", transform=ax.transAxes, ha='center', va='center')

        for gap in collect_gaps_between_blocks:
            rect = Rectangle((gap[0],  ylim[0] + (ylim[1] - ylim[0])*0.02), gap[1], (ylim[1] - ylim[0])*0.96, fill=None, edgecolor='r', linewidth=2, alpha=.5)
            ax.add_patch(rect)
    else:
        ax.text(0.5, 0.8, "No gaps found", transform=ax.transAxes, ha='center',
                va='center')

    sns.despine(ax=ax)
    return trials


def select_DIO_sync_trial_trace(path_recording_folder, rec_file_name):
    '''We select from the DIO trace containing TTL pulses only the trial starts (the ones)'''
    path_DIO_folder = Path(path_recording_folder, f"{rec_file_name[:rec_file_name.rfind('.')]}.DIO")
    DIO_dict = select_DIO_channel(path_DIO_folder)
    # Each data point is (timestamp, state) -> break into separate arrays
    DIO_data = DIO_dict['data'].copy()
    DIO_states = np.array([tup[1] for tup in DIO_data])
    DIO_samples = np.array([tup[0] for tup in DIO_data])
    DIO_timestamps = np.array([tup[0] for tup in DIO_data])/float(DIO_dict['clockrate'])
    assert DIO_states.shape == DIO_timestamps.shape
    DIO_timestamps_start_trial = DIO_timestamps[DIO_states.astype(bool)].copy()
    DIO_samples_start_trial = DIO_samples[DIO_states.astype(bool)].copy()

    return DIO_timestamps_start_trial, DIO_samples_start_trial

def Trim_TTLs(trials, DIO_timestamps_start_trial, DIO_samples_start_trial, min_distances):
    if len(trials)!=len(DIO_timestamps_start_trial):
        print("unequal numbers of trials between bpod and DIO")
        if np.argmax(min_distances) == len(trials):
            print("One extra TTL pulse received on DIO at the end of the session")
            DIO_timestamps_start_trial = DIO_timestamps_start_trial[:-1]
            DIO_samples_start_trial =  DIO_samples_start_trial[:-1]
            print("extra TTL pulse removed")
    return DIO_timestamps_start_trial,  DIO_samples_start_trial

def assign_DIO_times_to_trials(trials, DIO_timestamps_start_trial, DIO_samples_start_trial):
    fig, ax = plt.subplots()
    ax.plot(DIO_timestamps_start_trial, trials["bpod_start_time"])
    ax.set_xlabel("DIO trials start time (s)")
    ax.set_ylabel("Bpod trials start time (s)")
    ax.set_title("Should be the straightest line")
    trials["DIO_start_sample"] = DIO_samples_start_trial
    trials["DIO_start_time"] = DIO_timestamps_start_trial
    trials["DIO_start_sample_zeroed"] = trials["DIO_start_sample"] - trials["DIO_start_sample"][0]
    return trials


def plot_probe(raw_rec, channel_labels):
    y_lim_widget = pn.widgets.EditableRangeSlider(
        name='y_lim', start=0, end=raw_rec.get_channel_locations().max(),
        value=(raw_rec.get_channel_locations().max() - 800, raw_rec.get_channel_locations().max() - 200),
        step=10)

    channels_colors = [channel_label_color_dict[label] for label in channel_labels]

    @pn.depends(y_lim_widget)
    def inspect_probes_channels_labels(ylim):
        fig, axs = plt.subplots(1, 3, figsize=(10, 6))

        sw.plot_probe_map(raw_rec, color_channels=channels_colors, ax=axs[0], with_channel_ids=False)

        sw.plot_probe_map(raw_rec, color_channels=channels_colors, ax=axs[1], with_channel_ids=False)

        patches = [mpatches.Patch(color=color, label=label) for label, color in channel_label_color_dict.items()]

        axs[2].legend(handles=patches, loc='upper left', frameon=False);
        axs[2].axis("off");
        # axs[0].

        axs[1].set_ylim(ylim[0], ylim[1])

        # Draw a rectangle on axs[0] with these ylims
        # Assuming arbitrary x values, here 0 to 10 for illustration
        rect = mpatches.Rectangle((-100, ylim[0]), 600, ylim[1] - ylim[0], linewidth=1, edgecolor='r', facecolor='none')

        axs[0].add_patch(rect)
        plt.close()
        return pn.pane.Matplotlib(fig)

    return pn.Column(y_lim_widget, inspect_probes_channels_labels)

