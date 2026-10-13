| config | granted choice, step ms | SC-4 choice, step ms | SC-4 runner-up, ms | margin % (granted → SC-4) | tie | refused (granted → SC-4) | choice changed |
|---|---|---|---|---|---|---|---|
| `32xH100-400G-1gph-k10` | S 16, w 2, pieces + native, 1136.3 | **S 16, w 2, zero1 + native**, 1232.1 | S 32, w 1 (32 GPUs), 1316.0 | 2.41 → 6.81 | False → False | 2 → 3 | **yes** |
| `32xH100-400G-2gph-k10` | S 16, w 2, zero1 + native, 1163.7 | **S 32, w 1 (32 GPUs)**, 1316.0 | S 16, w 2, zero1 + native, 1394.3 | 1.06 → 5.95 | True → False | 3 → 3 | **yes** |
| `32xH100-400G-4gph-k10` | S 32, w 1 (32 GPUs), 1275.4 | **S 32, w 1 (32 GPUs)**, 1316.0 | S 16, w 2, zero1 + native, 1830.4 | 6.26 → 39.08 | False → False | 3 → 3 | no |
| `32xH100-400G-8gph-k10` | S 32, w 1 (32 GPUs), 1275.4 | **S 32, w 1 (32 GPUs)**, 1316.0 | S 16, w 2, zero1 + native, 2700.5 | 89.04 → 105.2 | False → False | 3 → 3 | no |
| `32xH100-400G-8gph-k2` | S 16, w 2, pieces + native, 1136.3 | **S 16, w 2, pieces + native**, 1155.4 | S 16, w 2, zero1 + native, 1230.3 | 2.41 → 6.48 | False → False | 2 → 2 | no |
| `32xH100-100G-1gph-k10` | S 32, w 1 (32 GPUs), 1275.4 | **S 32, w 1 (32 GPUs)**, 1440.8 | S 16, w 2, zero1 + native, 2471.5 | 65.27 → 71.53 | False → False | 3 → 3 | no |
| `32xH100-100G-8gph-k10` | S 32, w 1 (32 GPUs), 1275.4 | **S 32, w 1 (32 GPUs)**, 1440.8 | S 16, w 2, zero1 + native, 3199.5 | 89.04 → 122.06 | False → False | 3 → 3 | no |
| `128xH100-400G-1gph-k10` | S 32, w 4, zero1 + native, 1808.6 | **S 32, w 4, zero1 + native**, 2270.2 | S 32, w 2, pieces + native, 2358.7 | 15.0 → 3.9 | False → False | 2 → 2 | no |
| `128xH100-400G-1gph-k6` | S 32, w 4, pieces + native, 1297.2 | **S 32, w 4, pieces + native**, 1338.1 | S 32, w 4, zero1 + native, 2209.6 | 39.42 → 65.12 | False → False | 1 → 1 | no |
| `128xH100-400G-1gph-k2` | S 32, w 4, pieces + native, 1297.2 | **S 32, w 4, pieces + native**, 1338.1 | S 32, w 4, zero1 + native, 2209.6 | 39.42 → 65.12 | False → False | 1 → 1 | no |
| `128xH100-400G-8gph-k10` | S 32, w 1 (32 GPUs), 4324.8 | **S 32, w 1 (32 GPUs)**, 4365.4 | S 32, w 4, zero1 + native, 5305.6 | 16.5 → 21.54 | False → False | 2 → 3 | no |
| `128xH100-400G-8gph-k2` | S 32, w 4, zero1 + native, 1808.6 | **S 32, w 4, zero1 + native**, 2245.9 | S 32, w 2, pieces + native, 2344.5 | 15.0 → 4.39 | False → False | 2 → 2 | no |
| `128xH100-100G-1gph-k10` | S 32, w 1 (32 GPUs), 4324.8 | **S 32, w 1 (32 GPUs)**, 4490.3 | S 32, w 2, zero1 + native, 5709.0 | 44.74 → 27.14 | False → False | 2 → 3 | no |
| `128xH100-100G-8gph-k10` | S 32, w 1 (32 GPUs), 4324.8 | **S 32, w 1 (32 GPUs)**, 4490.3 | S 32, w 2, zero1 + native, 7601.9 | 44.74 → 69.3 | False → False | 2 → 3 | no |
| `32xH100-2sites-10G-uplink` | S 16, w 2, pieces + native, 1293.7 | **S 16, w 2, zero1 + native**, 1388.2 | S 32, w 1 (32 GPUs), 1418.4 | 2.12 → 2.18 | False → False | 5 → 6 | **yes** |
| `32xGPU-mixed-classes` | S 32, w 1 (32 GPUs), 1275.4 | **S 32, w 1 (32 GPUs)**, 1316.0 | — | None → None | False → False | 6 → 6 | no |
| `32xH100-400G-1gph-k10-memhost20` | S 32, w 1 (32 GPUs), 1275.4 | **S 16, w 2, zero1 + native**, 1232.1 | S 32, w 1 (32 GPUs), 1316.0 | None → 6.81 | False → False | 6 → 3 | **yes** |

Refusals pinned: 48 → 50.

| config | refused candidate | reasons (kind, where, value vs bound) |
|---|---|---|
| `32xH100-400G-1gph-k10` | S 16 w 2 pieces | S2 at stage 0 (first): 32.18 vs 23.63 ms |
| `32xH100-400G-1gph-k10` | S 8 w 4 pieces | S2 at stage 0 (first): 77.8 vs 21.05 ms |
| `32xH100-400G-1gph-k10` | S 4 w 8 pieces | S2 at stage 0 (first): 154.18 vs 19.76 ms |
| `32xH100-400G-2gph-k10` | S 16 w 2 pieces | S1 host DRAM at host 0 (stages [0]): 760.8 vs 600.0 GB/s; S2 at stage 0 (first): 202.68 vs 23.63 ms |
| `32xH100-400G-2gph-k10` | S 8 w 4 pieces | S1 host DRAM at host 0 (stages [0]): 875.7 vs 600.0 GB/s; S2 at stage 0 (first): 478.9 vs 21.05 ms |
| `32xH100-400G-2gph-k10` | S 4 w 8 pieces | S1 host DRAM at host 0 (stages [0]): 875.6 vs 600.0 GB/s; S2 at stage 0 (first): 656.92 vs 19.76 ms |
| `32xH100-400G-4gph-k10` | S 16 w 2 pieces | S1 host DRAM at host 0 (stages [0, 1]): 1249.0 vs 600.0 GB/s; S2 at stage 0 (first): 638.76 vs 23.63 ms |
| `32xH100-400G-4gph-k10` | S 8 w 4 pieces | S1 host DRAM at host 0 (stages [0]): 1751.5 vs 600.0 GB/s; S2 at stage 0 (first): 1469.76 vs 21.05 ms |
| `32xH100-400G-4gph-k10` | S 4 w 8 pieces | S1 host DRAM at host 0 (stages [0]): 1751.2 vs 600.0 GB/s; S2 at stage 0 (first): 1768.93 vs 19.76 ms |
| `32xH100-400G-8gph-k10` | S 16 w 2 pieces | S1 host DRAM at host 0 (stages [0, 1, 2, 3]): 2225.6 vs 600.0 GB/s; S2 at stage 0 (first): 1508.87 vs 23.63 ms |
| `32xH100-400G-8gph-k10` | S 8 w 4 pieces | S1 host DRAM at host 0 (stages [0, 1]): 3076.8 vs 600.0 GB/s; S2 at stage 0 (first): 2878.44 vs 21.05 ms |
| `32xH100-400G-8gph-k10` | S 4 w 8 pieces | S1 host DRAM at host 0 (stages [0]): 3502.4 vs 600.0 GB/s; S2 at stage 0 (first): 3992.93 vs 19.76 ms |
| `32xH100-400G-8gph-k2` | S 8 w 4 pieces | S1 host DRAM at host 0 (stages [0, 1]): 615.4 vs 600.0 GB/s; S2 at stage 0 (first): 142.07 vs 21.05 ms |
| `32xH100-400G-8gph-k2` | S 4 w 8 pieces | S1 host DRAM at host 0 (stages [0]): 700.5 vs 600.0 GB/s; S2 at stage 0 (first): 434.51 vs 19.76 ms |
| `32xH100-100G-1gph-k10` | S 16 w 2 pieces | S1 NIC at first stage: 1.522 vs 1.0 x cap*c_s; S2 at stage 0 (first): 361.15 vs 23.63 ms |
| `32xH100-100G-1gph-k10` | S 8 w 4 pieces | S1 NIC at first stage: 1.751 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.325 vs 1.0 x cap*c_s; S1 NIC at last stage: 1.312 vs 1.0 x cap*c_s; S2 at stage 0 (first): 677.12 vs 21.05 ms |
| `32xH100-100G-1gph-k10` | S 4 w 8 pieces | S1 NIC at first stage: 1.751 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.5 vs 1.0 x cap*c_s; S1 NIC at last stage: 1.497 vs 1.0 x cap*c_s; S2 at stage 0 (first): 879.33 vs 19.76 ms |
| `32xH100-100G-8gph-k10` | S 16 w 2 pieces | S1 NIC at first stage: 1.522 vs 1.0 x cap*c_s; S1 host DRAM at host 0 (stages [0, 1, 2, 3]): 2225.6 vs 600.0 GB/s; S2 at stage 0 (first): 1575.4 vs 23.63 ms |
| `32xH100-100G-8gph-k10` | S 8 w 4 pieces | S1 NIC at first stage: 1.751 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.325 vs 1.0 x cap*c_s; S1 NIC at last stage: 1.312 vs 1.0 x cap*c_s; S1 host DRAM at host 0 (stages [0, 1]): 3076.8 vs 600.0 GB/s; S2 at stage 0 (first): 2876.48 vs 21.05 ms |
| `32xH100-100G-8gph-k10` | S 4 w 8 pieces | S1 NIC at first stage: 1.751 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.5 vs 1.0 x cap*c_s; S1 NIC at last stage: 1.497 vs 1.0 x cap*c_s; S1 host DRAM at host 0 (stages [0]): 3502.4 vs 600.0 GB/s; S2 at stage 0 (first): 3992.96 vs 19.76 ms |
| `128xH100-400G-1gph-k10` | S 32 w 4 pieces | S1 host DRAM at host 0 (stages [0]): 809.8 vs 600.0 GB/s; S2 at stage 0 (first): 96.28 vs 28.78 ms |
| `128xH100-400G-1gph-k10` | S 16 w 8 pieces | S1 host DRAM at host 0 (stages [0]): 652.6 vs 600.0 GB/s; S2 at stage 0 (first): 103.7 vs 23.63 ms |
| `128xH100-400G-1gph-k6` | S 16 w 8 pieces | S2 at stage 0 (first): 34.4 vs 23.63 ms |
| `128xH100-400G-1gph-k2` | S 16 w 8 pieces | S2 at stage 0 (first): 34.4 vs 23.63 ms |
| `128xH100-400G-8gph-k10` | S 32 w 2 pieces | S1 host DRAM at host 0 (stages [0, 1, 2, 3]): 2777.1 vs 600.0 GB/s; S2 at stage 0 (first): 3563.98 vs 57.56 ms |
| `128xH100-400G-8gph-k10` | S 32 w 4 pieces | S1 host DRAM at host 0 (stages [0, 1]): 4773.8 vs 600.0 GB/s; S2 at stage 0 (first): 3875.29 vs 28.78 ms |
| `128xH100-400G-8gph-k10` | S 16 w 8 pieces | S1 host DRAM at host 0 (stages [0]): 5220.7 vs 600.0 GB/s; S2 at stage 0 (first): 4951.38 vs 23.63 ms |
| `128xH100-400G-8gph-k2` | S 32 w 4 pieces | S1 host DRAM at host 0 (stages [0, 1]): 954.8 vs 600.0 GB/s; S2 at stage 0 (first): 189.95 vs 28.78 ms |
| `128xH100-400G-8gph-k2` | S 16 w 8 pieces | S1 host DRAM at host 0 (stages [0]): 1044.1 vs 600.0 GB/s; S2 at stage 0 (first): 519.21 vs 23.63 ms |
| `128xH100-100G-1gph-k10` | S 32 w 2 pieces | S1 NIC at first stage: 2.206 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.116 vs 1.0 x cap*c_s; S2 at stage 0 (first): 901.57 vs 57.56 ms |
| `128xH100-100G-1gph-k10` | S 32 w 4 pieces | S1 NIC at first stage: 3.239 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.535 vs 1.0 x cap*c_s; S1 NIC at last stage: 1.42 vs 1.0 x cap*c_s; S1 host DRAM at host 0 (stages [0]): 809.8 vs 600.0 GB/s; S2 at stage 0 (first): 1244.43 vs 28.78 ms |
| `128xH100-100G-1gph-k10` | S 16 w 8 pieces | S1 NIC at first stage: 2.61 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.604 vs 1.0 x cap*c_s; S1 NIC at last stage: 1.573 vs 1.0 x cap*c_s; S1 host DRAM at host 0 (stages [0]): 652.6 vs 600.0 GB/s; S2 at stage 0 (first): 1073.25 vs 23.63 ms |
| `128xH100-100G-8gph-k10` | S 32 w 2 pieces | S1 NIC at first stage: 2.206 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.116 vs 1.0 x cap*c_s; S1 host DRAM at host 0 (stages [0, 1, 2, 3]): 2777.1 vs 600.0 GB/s; S2 at stage 0 (first): 3696.72 vs 57.56 ms |
| `128xH100-100G-8gph-k10` | S 32 w 4 pieces | S1 NIC at first stage: 3.239 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.535 vs 1.0 x cap*c_s; S1 NIC at last stage: 1.42 vs 1.0 x cap*c_s; S1 host DRAM at host 0 (stages [0, 1]): 4773.8 vs 600.0 GB/s; S2 at stage 0 (first): 3873.33 vs 28.78 ms |
| `128xH100-100G-8gph-k10` | S 16 w 8 pieces | S1 NIC at first stage: 2.61 vs 1.0 x cap*c_s; S1 NIC at mid stage: 1.604 vs 1.0 x cap*c_s; S1 NIC at last stage: 1.573 vs 1.0 x cap*c_s; S1 host DRAM at host 0 (stages [0]): 5220.7 vs 600.0 GB/s; S2 at stage 0 (first): 4951.41 vs 23.63 ms |
| `32xH100-2sites-10G-uplink` | S 16 w 2 pieces | S2 at stage 0 (first): 32.18 vs 23.63 ms |
| `32xH100-2sites-10G-uplink` | S 16 w 2 pieces, straddling | S1 uplink at site0-site1: 175.5 vs 1.25 GB/s |
| `32xH100-2sites-10G-uplink` | S 8 w 4 pieces | S2 at stage 0 (first): 77.8 vs 21.05 ms |
| `32xH100-2sites-10G-uplink` | S 8 w 4 pieces, straddling | S1 uplink at site0-site1: 175.5 vs 1.25 GB/s |
| `32xH100-2sites-10G-uplink` | S 4 w 8 pieces | S2 at stage 0 (first): 154.18 vs 19.76 ms |
| `32xH100-2sites-10G-uplink` | S 4 w 8 pieces, straddling | S1 uplink at site0-site1: 175.5 vs 1.25 GB/s |
| `32xGPU-mixed-classes` | S 16 w 2 pieces | class at stage 8: A+B vs one class  |
| `32xGPU-mixed-classes` | S 16 w 2 zero1 + native | class at stage 8: A+B vs one class  |
| `32xGPU-mixed-classes` | S 8 w 4 pieces | class at stage 4: A+B vs one class  |
| `32xGPU-mixed-classes` | S 8 w 4 zero1 + native | class at stage 4: A+B vs one class  |
| `32xGPU-mixed-classes` | S 4 w 8 pieces | class at stage 2: A+B vs one class  |
| `32xGPU-mixed-classes` | S 4 w 8 zero1 + native | class at stage 2: A+B vs one class  |
| `32xH100-400G-1gph-k10-memhost20` | S 16 w 2 pieces | S2 at stage 0 (first): 32.18 vs 23.63 ms; MEM-HOST at host of a first (2 blocks) peer: 23.74 vs 20.0 GB |
| `32xH100-400G-1gph-k10-memhost20` | S 8 w 4 pieces | S2 at stage 0 (first): 77.8 vs 21.05 ms; MEM-HOST at host of a first (4 blocks) peer: 29.73 vs 20.0 GB |
| `32xH100-400G-1gph-k10-memhost20` | S 4 w 8 pieces | S2 at stage 0 (first): 154.18 vs 19.76 ms; MEM-HOST at host of a first (8 blocks) peer: 33.36 vs 20.0 GB |

| config | SC-4 (spec) | comm0 | nodram | embed_p0 (diagnostic) |
|---|---|---|---|---|
| `32xH100-400G-1gph-k10` | S 16 w 2 zero1 + native 1232.1 | S 16 w 2 zero1 + native 1212.0 (+5.15%) | S 16 w 2 pieces + native 1155.4 (+6.48%) | S 16 w 2 zero1 + native 1233.0 (+6.73%) |
| `32xH100-400G-2gph-k10` | S 32 w 1 1316.0 | S 32 w 1 1274.4 (+7.83%) | S 16 w 2 zero1 + native 1230.3 (+6.96%) | S 32 w 1 1316.0 (+5.95%) |
| `32xH100-400G-4gph-k10` | S 32 w 1 1316.0 | S 32 w 1 1274.4 (+42.1%) | S 16 w 2 zero1 + native 1230.3 (+6.96%) | S 32 w 1 1316.0 (+39.08%) |
| `32xH100-400G-8gph-k10` | S 32 w 1 1316.0 | S 32 w 1 1274.4 (+110.48%) | S 16 w 2 zero1 + native 1230.3 (+6.96%) | S 32 w 1 1316.0 (+105.2%) |
| `32xH100-400G-8gph-k2` | S 16 w 2 pieces + native 1155.4 | S 16 w 2 pieces + native 1135.3 (+6.6%) | S 16 w 2 pieces + native 1155.4 (+6.48%) | S 16 w 2 pieces + native 1155.4 (+6.71%) |
| `32xH100-100G-1gph-k10` | S 32 w 1 1440.8 | S 32 w 1 1274.4 (+87.61%) | S 32 w 1 1440.8 (+71.53%) | S 32 w 1 1440.8 (+77.86%) |
| `32xH100-100G-8gph-k10` | S 32 w 1 1440.8 | S 32 w 1 1274.4 (+145.08%) | S 32 w 1 1440.8 (+71.53%) | S 32 w 1 1440.8 (+125.91%) |
| `128xH100-400G-1gph-k10` | S 32 w 4 zero1 + native 2270.2 | S 32 w 4 zero1 + native 2228.6 (+3.97%) | S 32 w 4 zero1 + native 2209.6 (+6.11%) | S 32 w 2 pieces + native 2358.7 (+0.88%) |
| `128xH100-400G-1gph-k6` | S 32 w 4 pieces + native 1338.1 | S 32 w 4 pieces + native 1296.5 (+67.21%) | S 32 w 4 pieces + native 1338.1 (+65.12%) | S 32 w 4 pieces + native 1338.1 (+75.05%) |
| `128xH100-400G-1gph-k2` | S 32 w 4 pieces + native 1338.1 | S 32 w 4 pieces + native 1296.5 (+67.21%) | S 32 w 4 pieces + native 1338.1 (+65.12%) | S 32 w 4 pieces + native 1338.1 (+75.05%) |
| `128xH100-400G-8gph-k10` | S 32 w 1 4365.4 | S 32 w 1 4323.8 (+21.76%) | S 32 w 4 zero1 + native 2209.6 (+12.19%) | S 32 w 1 4365.4 (+21.54%) |
| `128xH100-400G-8gph-k2` | S 32 w 4 zero1 + native 2245.9 | S 32 w 4 zero1 + native 2213.2 (+4.05%) | S 32 w 4 zero1 + native 2209.6 (+6.11%) | S 32 w 4 zero1 + native 2344.0 (+0.02%) |
| `128xH100-100G-1gph-k10` | S 32 w 1 4490.3 | S 32 w 1 4323.8 (+28.19%) | S 32 w 1 4490.3 (+27.14%) | S 32 w 1 4490.3 (+34.91%) |
| `128xH100-100G-8gph-k10` | S 32 w 1 4490.3 | S 32 w 1 4323.8 (+75.77%) | S 32 w 1 4490.3 (+27.14%) | S 32 w 1 4490.3 (+73.88%) |
| `32xH100-2sites-10G-uplink` | S 16 w 2 zero1 + native 1388.2 | S 16 w 2 zero1 + native 1369.4 (+0.64%) | S 16 w 2 pieces + native 1311.5 (+5.71%) | S 16 w 2 zero1 + native 1389.1 (+2.11%) |
| `32xGPU-mixed-classes` | S 32 w 1 1316.0 | S 32 w 1 1274.4 (+None%) | S 32 w 1 1316.0 (+None%) | S 32 w 1 1316.0 (+None%) |
| `32xH100-400G-1gph-k10-memhost20` | S 16 w 2 zero1 + native 1232.1 | S 16 w 2 zero1 + native 1212.0 (+5.15%) | S 16 w 2 zero1 + native 1230.3 (+6.96%) | S 16 w 2 zero1 + native 1233.0 (+6.73%) |
