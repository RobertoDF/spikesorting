from Process.process import process_trials, spikesort
from tqdm import tqdm
from colored import Fore, Back, Style

sessions = [r"/alzheimer/Roberto/Dariya/12/ephys/20231210_191835.rec/",
            r"/alzheimer/Roberto/Dariya/12/ephys/20240126_184212.rec/",
            r"/alzheimer/Roberto/Dariya/12/ephys/20231212_183552.rec/"]


n=0

for session in tqdm(sessions):
    try:
        process_trials(session)
        spikesort(session)
        n=+1
    except Exception as e:
        print(f"{Fore.white}{Back.red}{e}{Style.reset}")

print(f"Processed succesfully {n} of {len(sessions)} sessions")

# after open spikesorting_single_sesssion notebook andgo to "Open phy" section for manual curation

