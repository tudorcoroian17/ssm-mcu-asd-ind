# Phase 1. GPU Prototype
## Section 1.1 Dataset Inventory

Answers for the verification gate:

| Item | Question                                         | Answer                                                                                                                                                                |
|------|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | Uniform sample rate?                             | Yes. All `.wav` file are sampled at `48 kHz`                                                                                                                          |
| 2    | Uniform clip duration?                           | Uniform withing subsets. `IND` and anomalous subsets are `11 s` long.                                                                                                 |
| 3    | Channel count                                    | All clips have only one channel. Each `case` folder has 4 individual channel files for each recorded event. Only events picked up on channel 1 (`ch1`) where selected |

### Clip counts per `(machine_type, case_id, label)` with additional `source` - only channel 1 (`ch1`)

| Machine type | Case ID | Label   | Source | Count of clips |
|--------------|---------|---------|--------|----------------|
| ToyCar       | 1       | anomaly | IND    | 264            |
| ToyCar       | 1       | normal  | IND    | 1350           |                 
| ToyCar       | 2       | anomaly | IND    | 265            |
| ToyCar       | 2       | normal  | IND    | 1350           |
| ToyCar       | 3       | anomaly | IND    | 265            |
| ToyCar       | 3       | normal  | IND    | 1350           |
| ToyCar       | 4       | anomaly | IND    | 265            |
| ToyCar       | 4       | normal  | IND    | 1350           |
