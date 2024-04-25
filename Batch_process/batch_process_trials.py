from Process.process import process_trials
from tqdm import tqdm

sessions = [r"/alzheimer/Roberto/Dariya/12/ephys/202312142_183552.rec/",
            r"/alzheimer/Roberto/Dariya/12/ephys/20240126_184212.rec/",
            r"/alzheimer/Roberto/Dariya/12/ephys/20231210_191835.rec/"]


n=0
for session in tqdm(sessions):
    try:
        process_trials(session)
        n=+1
    except Exception as e:
        print(e)

print(f"Processed succesfully {n} of {len(sessions)} sessions")