# Ott lab ephys (spikegadgets+Neuropixels) data processing

* spike_times are extracted using spikeinterface
* units info are stored in a dataframe called units.csv
* trial info are stored in a dataframe called trials.csv

## Pipeline is adapted to multiple Neuropixels recording + Bpod 
It automatically pairs recordings to bpod files.

### Imperative folder structure: 
- /n_animal
  - /ephys 
    - 20240126_184212.rec
    - 20240221_184222.rec
  - /bpod_session 
    - 20240126_184212
    - 20240221_184212
    - 20240225_184317
    - 20240227_184548

## Visualizations notebook contains a interactive PSTH
![PSTH](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdWQxbHBnYnl4NmR5a2JvY2c2MGlyd2g0MHI0OHhlNm55d3Y3MHg0cSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/wA5wMVKxnZVCVIAyGm/giphy.gif)

### Change Utils.Paths if needed, computer specific!
