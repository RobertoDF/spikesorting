# Ott lab ephys (spikegadgets+Neuropixels) data processing

* spike_times are extracted using spikeinterface
* units info are stored in a dataframe called units.csv
* trial info are stored in a dataframe called trials.csv

## Pipeline is adapted to multiple Neuropixels recording + Bpod 
It automatically pairs recordings to bpod files.

### Imperative folder structure: n_animal/
- /n_animal
  - /ephys (it has to be called like this)
    - 20240126_184212.rec
    - 20240221_184222.rec
  - /bpod_session (it has to be called like this)
    - 20240126_184212
    - 20240221_184212
