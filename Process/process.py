
from spikeinterface.extractors import read_spikegadgets
from spikeinterface.preprocessing import detect_bad_channels
from tqdm.notebook import tqdm
from colored import Fore, Back, Style
from pathlib import Path
import torch
from Utils.Utils import add_custom_metrics_to_phy_folder, check_gpu_availability, clean_trials, get_timestamps_from_rec, get_recording_time, assign_DIO_times_to_trials, \
    Trim_TTLs, select_DIO_sync_trial_trace, stitch_bpod_times, find_min_distance_TTL, call_trodesexport, \
    check_single_rec_file, check_timestamps_gaps, get_mouse_name, get_recording_day, find_mat_files_with_same_day
from Utils.Utils import print_in_color
import matplotlib.pyplot as plt
import os
from spikeinterface.sorters import run_sorter
import pandas as pd
import numpy as np
from Utils.Settings import channel_label_color_dict
from spikeinterface.widgets import plot_probe_map

check_gpu_availability()
# # **Ott lab process single session**
# #####  Multi-Neuropixels recording using SpikeGadgets + Bpod

# ## Imperative folder structure: n_animal/

# - /n_animal
#   - /ephys (it has to be called like this)
#     - 20240126_184212.rec
#     - 20240221_184222.rec
#   - /bpod_session (it has to be called like this)
#     - 20240126_184212
#     - 20240221_184212

# # Select file

def process_trials(path):
    path_recording_folder = Path(path)

    print(f'{Fore.white}{Back.green}Processing session {path_recording_folder.name}{Style.reset}')

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Session {path_recording_folder.name}", fontsize=16)

    axs = axs.ravel()

    mouse_n = get_mouse_name(path_recording_folder)
    day = get_recording_day(path_recording_folder)
    time = get_recording_time(path_recording_folder)
    print(f"mouse {mouse_n} recorded on {day} at {time}")

    path_recording, rec_file_name = check_single_rec_file(path_recording_folder)

    # ## Extract timestamps from .rec file

    timestamps = get_timestamps_from_rec(path_recording_folder,  path_recording)

    # ## Load recording in spikeinterface

    raw_rec = read_spikegadgets(path_recording)
    fs = raw_rec.get_sampling_frequency()
    correct_times = timestamps/fs
    raw_rec.set_times(correct_times)  # set new times

    print(f"Recording duration in minutes: {raw_rec.get_total_duration() / 60}, sampling rate: {fs} Hz")
    print(f"Probes present: {raw_rec.get_probes()}")


    gaps_start_stop = check_timestamps_gaps(raw_rec, correct_times)

    # # Sync Bpod and Trodes streams
    # ### Export Digital IO channels

    call_trodesexport(path_recording_folder, path_recording, "dio")

    DIO_timestamps_start_trial, DIO_samples_start_trial = select_DIO_sync_trial_trace(path_recording_folder, rec_file_name)

    # ## Load bpod mat file behavior

    bpod_file = find_mat_files_with_same_day(path_recording_folder.parent.parent / "bpod_session" , path_recording_folder, raw_rec)

    # ## Stitch trials and trim as needed

    trials =  stitch_bpod_times(bpod_file, day, DIO_timestamps_start_trial,axs[0])

    min_distances = find_min_distance_TTL(DIO_timestamps_start_trial,  trials["bpod_start_time"], axs[1])

    # only adapted to case where there is a extra TTL in the DIO at the end of last stimulus block!
    DIO_timestamps_start_trial,  DIO_samples_start_trial = Trim_TTLs(trials, DIO_timestamps_start_trial, DIO_samples_start_trial, min_distances)


    min_distances = find_min_distance_TTL(DIO_timestamps_start_trial,  trials["bpod_start_time"], axs[2])

    trials = assign_DIO_times_to_trials(trials, DIO_timestamps_start_trial, DIO_samples_start_trial, axs[3])

    # ## Final Trial df

    cleaned_trials = clean_trials(trials, raw_rec, gaps_start_stop)

    cleaned_trials.to_csv(Path(f"{path_recording_folder}/trials.csv"))

    print_in_color(f"Session {path_recording_folder.name} processed","green")

    plt.tight_layout()
    plt.show()


def spikesort(path):
    path_recording_folder = Path(path)

    fig, axs = plt.subplots(1, 1, figsize=(15, 10))
    fig.suptitle(f"Session {path_recording_folder.name}", fontsize=16)

    print(f'{Fore.white}{Back.green}Spikesorting session {path_recording_folder.name}{Style.reset}')

    mouse_n = get_mouse_name(path_recording_folder)
    day = get_recording_day(path_recording_folder)
    time = get_recording_time(path_recording_folder)
    print(f"mouse {mouse_n} recorded on {day} at {time}")

    path_recording, rec_file_name = check_single_rec_file(path_recording_folder)

    timestamps = get_timestamps_from_rec(path_recording_folder, path_recording)
    raw_rec = read_spikegadgets(path_recording)

    fs = raw_rec.get_sampling_frequency()

    correct_times = timestamps / fs
    raw_rec.set_times(correct_times)  # set new times

    print(f"Recording duration in minutes: {raw_rec.get_total_duration() / 60}, sampling rate: {fs} Hz")
    print(f"Probes present: {raw_rec.get_probes()}")

    if os.path.exists(f"{path_recording_folder}/channel_labels.csv"):
        channel_labels = pd.read_csv(f"{path_recording_folder}/channel_labels.csv")
    else:
        bad_channel_ids_list = []
        channel_labels_list = []
        # detect noisy, dead, and out-of-brain channels
        split_preprocessed_recording = raw_rec.split_by("group")
        for group, sub_rec in tqdm(split_preprocessed_recording.items()):
            bad_channel_ids, channel_labels = detect_bad_channels(sub_rec)

            bad_channel_ids_list.append(bad_channel_ids)
            channel_labels_list.extend(channel_labels)

        channel_labels = pd.DataFrame([channel_labels_list], index=["channel_labels"])
        channel_labels.to_csv(f"{path_recording_folder}/channel_labels.csv", index=False)

    bad_channel_ids = channel_labels[~(channel_labels["channel_labels"] == "good")]

    print(channel_labels["channel_labels"].value_counts())

    raw_rec = raw_rec.remove_channels(bad_channel_ids)
    print(f"{len(bad_channel_ids)} bad channels removed")

    torch.cuda.empty_cache()

    split_preprocessed_recording = raw_rec.split_by("group")
    for group, sub_rec in split_preprocessed_recording.items():
        sorting = run_sorter(
            sorter_name="kilosort4",
            recording=sub_rec,
            output_folder=f"{path_recording_folder}/spike_interface_output/probe{group}",
            verbose=True,
            remove_existing_folder=True
        )

    add_custom_metrics_to_phy_folder(raw_rec, path_recording_folder)