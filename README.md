# Cross-Docking Simulation Input Dataset

**Source:** TORBALI (2023), _Real-Time Truck Scheduling in Cross-Docking: A Hybrid Multi-Agent Simulation Framework_

---

## 1. Overview

This dataset provides simulation input data for real-time truck scheduling and pallet flow analysis in a cross-docking terminal. It is designed to test and validate multi-agent and discrete-event simulation models for cross-dock operations. The data originates from the experiments described in Chapter 4 of the referenced thesis.

---

## 2. Folder Structure

The dataset is organised into six folders, each representing a combination of inbound and outbound traffic levels:

| Folder    | Inbound Traffic          | Outbound Traffic         | Duration | Meaning                             |
| :-------- | :----------------------- | :----------------------- | :------- | :---------------------------------- |
| `HH_168h` | High (156 pallets/hour)  | High (156 pallets/hour)  | 168 h    | High inbound and high outbound flow |
| `MH_168h` | Medium (78 pallets/hour) | High (156 pallets/hour)  | 168 h    | Medium inbound, high outbound       |
| `MM_168h` | Medium (78 pallets/hour) | Medium (78 pallets/hour) | 168 h    | Balanced medium flow                |
| `LH_168h` | Low (39 pallets/hour)    | High (156 pallets/hour)  | 168 h    | Low inbound, high outbound          |
| `LM_168h` | Low (39 pallets/hour)    | Medium (78 pallets/hour) | 168 h    | Low inbound, medium outbound        |
| `LL_168h` | Low (39 pallets/hour)    | Low (39 pallets/hour)    | 168 h    | Low inbound and low outbound flow   |

Each folder contains **10 Excel files** (`instance1.xlsx` … `instance10.xlsx`), representing **individual simulation instances**. All instances span the same time frame: **168 hours (7 days)**.

---

## 3. File Structure

Each Excel file contains **three worksheets**:

### 3.1 `inboundTrucks`

List of inbound trucks arriving during the 168 h simulation. Each row represents one truck.

| Column       | Type    | Description                                                                         |
| :----------- | :------ | :---------------------------------------------------------------------------------- |
| `TruckID`    | Integer | Unique identifier of the inbound truck.                                             |
| `ArrivalMin` | Integer | Truck arrival time in **minutes** since the start of the simulation (0–10 080 min). |

Inbound trucks deliver pallets to the cross-dock. The arrival process follows a **triangular distribution** for interarrival times depending on the traffic level (e.g., Triangular(7, 10, 13) min for high traffic).

---

### 3.2 `outboundTrucks`

List of outbound trucks scheduled to leave the cross-dock.

| Column        | Type          | Description                                                                                                  |
| :------------ | :------------ | :----------------------------------------------------------------------------------------------------------- |
| `TruckID`     | Integer       | Unique identifier of the outbound truck.                                                                     |
| `ArrivalMin`  | Integer       | Arrival (ready-to-load) time of the outbound truck in **minutes** since the start of the simulation.         |
| `DueMin`      | Integer       | Due time in **minutes** since start of simulation, representing the scheduled departure or service deadline. |
| `Destination` | Integer (1–3) | Destination site served by the truck.                                                                        |

Each outbound truck serves **one destination (1–3)** and has a **capacity of 26 pallets**.

---

### 3.3 `pallets`

List of all pallets handled during the simulation. Each row corresponds to one pallet.

| Column        | Type                | Description                                                            |
| :------------ | :------------------ | :--------------------------------------------------------------------- |
| `PalletID`    | Integer             | Unique identifier of the pallet.                                       |
| `TruckId`     | Integer             | The `TruckID` of the **inbound truck** that delivered this pallet.     |
| `DueMin`      | Integer             | Due time (deadline) in minutes for the pallet to leave the cross-dock. |
| `Destination` | Integer (1–3)       | Destination served by the pallet.                                      |
| `Type`        | Categorical (A/B/C) | Pallet type or product category.                                       |

Each pallet is initially part of an inbound truck’s load. The due date is generated as a **uniform random offset (60–1440 min)** after its inbound truck’s arrival. Pallet types (A, B, C) simulate product diversity and handling constraints.

---

## 4. Codes and Parameters

| Code                | Meaning                                 | Origin/Use                                           |
| :------------------ | :-------------------------------------- | :--------------------------------------------------- |
| `H`, `M`, `L`       | High, Medium, Low traffic levels        | Define pallet arrival rates per hour (156, 78, 39).  |
| `168h`              | Simulation time horizon                 | All instances cover 168 hours (7 days).              |
| `Destination` = 1–3 | Site served by outbound truck or pallet | Used to aggregate and analyse outbound flow.         |
| `Type` = A/B/C      | Pallet category                         | Used for load balancing and product-mix constraints. |

---

## 5. Statistical Distributions and Constants

| Parameter                   | Distribution                                                                                    | Description                                   |
| :-------------------------- | :---------------------------------------------------------------------------------------------- | :-------------------------------------------- |
| Inbound truck interarrival  | Triangular(7, 10, 13) min (H) / Triangular(17, 20, 23) min (M) / Triangular(37, 40, 43) min (L) | Controls inbound arrival frequency.           |
| Outbound truck interarrival | Same as inbound depending on traffic level                                                      | Determines outbound availability.             |
| Pallet due date offset      | Uniform(60, 1440) min                                                                           | Defines required departure window.            |
| Outbound due time offset    | Uniform(60, 180) min                                                                            | Defines truck service deadline.               |
| Truck capacity              | Constant = 26 pallets                                                                           | Each truck carries a full or partial load.    |
| Number of destinations      | 3                                                                                               | Represents three outbound distribution zones. |

---

## 6. Usage and Application

The dataset is intended for **optimisation and simulation studies** involving:

- Real-time truck scheduling
- Cross-dock door assignment
- Pallet flow coordination
- Service-level and delay analysis

Each file can be used to:

1. Aggregate inbound/outbound flows by day or hour.
2. Estimate fleet and dock capacity requirements.
3. Run discrete-event or multi-agent simulations to evaluate truck scheduling strategies.
4. Compare KPIs such as:
   - Average stock level
   - Number of late pallets
   - Outbound truck fill rate
   - Door utilisation

---

## 7. Summary

- **Total instances:** 60 (6 traffic-level folders × 10 files).
- **Total simulation time:** 168 hours per instance.
- **Model parameters:** Based on a single inbound and outbound door, 15 forklifts, 26 pallets per truck.
- **Purpose:** Provide realistic synthetic data for evaluating scheduling strategies under varying load conditions.

---

### References

Torbali, A. B. (2023). _Real-time Truck Scheduling in Cross-Docking: A Hybrid Multi-Agent Simulation Framework._ Université Grenoble Alpes.
Dataset documentation – Appendix D: Access to Input Datasets.
