# ACLED_Anomaly_Detection  
  
Original work: https://github.com/Ethan-Harris0n/ACLED_Anomaly_Detection
  
- Conflict_Spike_Detection.ipynb: Original Core Script
- Conflict_Spike_Nigeria_Webinar.ipynb: Is just the script used for the webinar - it does have some potentially useful code for generating anomaly detection at different admin levels / actor subsets but it needs to be standardized / optimized
### Next Steps ###

#### Primary ####
Optimize and convert the pasted functions used in webinar script intoto a replicable framework that can easily generate sub-anomalaly detection pipelines
for different user-specified admin levels and ACLED_Actor subsets


#### Secondary ####

- Evaluate how to apply to a conflict-dense profile (nigeria).
- Determine Potential Benefits / Downsides to:
  - Using Isolation Forests
  - HDB Scan Clusters (some good thoughts on cross validated about whether this is kosher).
  - Applying a simple auto-regressive process and using the resulting residuals to construct a band/threshhold for unexpected values.
  - or other methods
  
Goal should be to identify a reliable algorithm for constructing a threshold so as not to require (intensive) country-specific tuning.


