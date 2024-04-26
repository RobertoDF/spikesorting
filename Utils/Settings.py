import os


max_ISI_gap_recording = 0.001  # max intersample interval (ISI), above which the period is considered a "gap" in the recording


channel_label_color_dict = {
    "good": "#4CAF50",  # Green
    "dead": "#F44336",  # Red
    "noise": "#FFC107", # Amber
    "out": "#2196F3"   # Blue
}

trodesexport_flags_to_folder = {"analogio":".analog", "dio":".DIO", "kilosort": ".kilosort", "time": ".time"}

job_kwargs = dict(n_jobs=20, chunk_duration='30s', progress_bar=True)