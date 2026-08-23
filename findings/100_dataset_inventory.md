# Phase 1. GPU Prototype
## Section 1.1 Dataset Inventory

Answers for the verification gate:

| Item | Question                                         | Answer                                                                                                                                                                |
|------|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | Uniform sample rate?                             | Yes. All `.wav` file are sampled at `48 kHz`                                                                                                                          |
| 2    | Uniform clip duration?                           | Uniform withing subsets. `IND` and anomalous subsets are `11 s` long. `CNT` subsets are all `600 s` long                                                              |
| 3    | Channel count                                    | All clips have only one channel. Each `case` folder has 4 individual channel files for each recorded event. Only events picked up on channel 1 (`ch1`) where selected |

### Clip counts per `(machine_type, case_id, label)` with additional `source` - only channel 1 (`ch1`)

| Machine type | Case ID | Label   | Source | Count of clips | Silent Clips (`RMS < 0.001`) |
|--------------|---------|---------|--------|----------------|------------------------------|
| ToyCar       | 1       | anomaly | IND    | 264            | 16                           |
| ToyCar       | 1       | normal  | CNT    | 216            | 26                           |
| ToyCar       | 1       | normal  | IND    | 1350           | 14                           |                 
| ToyCar       | 2       | anomaly | IND    | 265            | 1                            |
| ToyCar       | 2       | normal  | CNT    | 147            | 27                           |
| ToyCar       | 2       | normal  | IND    | 1350           | 0                            |
| ToyCar       | 3       | anomaly | IND    | 265            | 5                            |
| ToyCar       | 3       | normal  | CNT    | 148            | 87                           |
| ToyCar       | 3       | normal  | IND    | 1350           | 8                            |
| ToyCar       | 4       | anomaly | IND    | 265            | 2                            |
| ToyCar       | 4       | normal  | CNT    | 160            | 77                           |
| ToyCar       | 4       | normal  | IND    | 1350           | 0                            |

### Silence contamination in CNT files

| case | whole-file estimate (old) | 	frame-level (real) |
|------|---------------------------|---------------------|
| 1    | 	12.0%                    | 	43.2%              |
| 2    | 	18.4%                    | 	40.2%              |
| 3    | 	58.8%                    | 	39.9%              |
| 4    | 	48.1%                    | 	33.1%              |

Check which LOSO fold has the smallest *clean* training pool

| Held-out | Trains on | Alive Windows | Total Windows | %     |
|----------|-----------|---------------|---------------|-------|
| 1        | 2, 3, 4   | 9,246         | 24,570        | 37.6% |
| 2        | 1, 3, 4   | 11,095        | 28,296        | 39.2% |
| 3        | 1, 2, 4   | 11,095        | 28,242        | 39.3% |
| 4        | 1, 2, 3   | 11,425        | 27,594        | 41.4% |