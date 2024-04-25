from spikeinterface.extractors import read_spikegadgets
from Utils.Utils import clean_trials, get_timestamps_from_rec, get_recording_time, assign_DIO_times_to_trials, \
    Trim_TTLs, select_DIO_sync_trial_trace, stitch_bpod_times, find_min_distance_TTL, call_trodesexport, \
    check_single_rec_file, check_timestamps_gaps, get_mouse_name, get_recording_day, find_mat_files_with_same_day
from pathlib import Path
import matplotlib.pyplot as plt
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
    path = Path(path)
    print(f"Processing session {path.name}")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Session {path.name}", fontsize=16)

    axs = axs.ravel()

    # folder containing .rec file
    path_recording_folder = path
    #path_recording_folder = Path(r"O:\data\12\ephys\20240126_184212.rec")

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
    DIO_timestamps_start_trial,  DIO_samples_start_trial = Trim_TTLs(trials, DIO_timestamps_start_trial, DIO_samples_start_trial,min_distances)


    min_distances = find_min_distance_TTL(DIO_timestamps_start_trial,  trials["bpod_start_time"], axs[2])

    trials = assign_DIO_times_to_trials(trials, DIO_timestamps_start_trial, DIO_samples_start_trial, axs[3])

    # ## Final Trial df

    cleaned_trials = clean_trials(trials, raw_rec, gaps_start_stop)

    cleaned_trials.to_csv(Path(f"{path_recording_folder}/trials.csv"))

    plt.tight_layout()
    plt.show()


