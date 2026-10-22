| run | envs | GPU placement | JAX alloc | client | wall [s] | batches | succ/attempted | success % | init→viser [s] | 1st batch [s] | batch median [s] | s / attempted traj | succ traj / h | GPU0 peak [GiB] | GPU1 Δ [GiB] | GPU0 util % | valid/saved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| run02_env2 | 2 | default (render+JAX on GPU1) | 25 % | chrome | 2400 | 129 | 222/258 | 86.0 | – | – | 13.0 | 6.50 | 456 | 9.7 | 6.1 | 17 | 222/222 |
| run04_env2_gpu0 | 2 | GPU0 | 25 % | chrome | 1201 | 92 | 160/184 | 87.0 | 15.6 | 43.4 | 12.4 | 6.22 | 495 | 14.4 | 0.0 | 19 | 160/160 |
| run05_env5_gpu0 | 5 | GPU0 | 25 % | chrome | 721 | 34 | 155/170 | 91.2 | 20.7 | 60.7 | 19.1 | 3.82 | 839 | 21.0 | 0.0 | 54 | 155/155 |
| run05_env10_gpu0 | 10 | GPU0 | 25 % | chrome | 780 | 0 | 0/0 | – | – | – | – | – | – | 23.5 | 0.0 | – | –/– |
| run06_env2_gpu0_noprealloc | 2 | GPU0 | lazy | chrome | 489 | 35 | 67/70 | 95.7 | 20.6 | 49.7 | 12.5 | 6.25 | 551 | 8.6 | 0.0 | 41 | 67/67 |
| run06_env5_gpu0_noprealloc | 5 | GPU0 | lazy | chrome | 480 | 22 | 97/110 | 88.2 | 16.9 | 51.0 | 19.1 | 3.83 | 797 | 14.0 | 0.1 | 54 | 97/97 |
| run06_env8_gpu0_noprealloc | 8 | GPU0 | lazy | chrome | 601 | 24 | 166/192 | 86.5 | 16.9 | 65.7 | 23.5 | 2.94 | 1078 | 16.7 | 0.0 | 70 | 166/166 |
| run06_env10_gpu0_noprealloc | 10 | GPU0 | lazy | chrome | 600 | 18 | 164/180 | 91.1 | 17.5 | 65.7 | 30.2 | 3.02 | 1065 | 19.3 | 0.0 | 63 | 164/164 |
| run06_env12_gpu0_noprealloc | 12 | GPU0 | lazy | chrome | 602 | 15 | 153/180 | 85.0 | 17.7 | 74.0 | 37.4 | 3.12 | 991 | 22.9 | 0.0 | 66 | 153/153 |
| run07_env2_gpu0_noprealloc_nullclient | 2 | GPU0 | lazy | null | 483 | 45 | 80/90 | 88.9 | 18.7 | 40.0 | 9.8 | 4.88 | 641 | 8.5 | 0.0 | 53 | 80/80 |
| run07_env5_gpu0_noprealloc_nullclient | 5 | GPU0 | lazy | null | 481 | 27 | 122/135 | 90.4 | 15.9 | 44.9 | 16.2 | 3.24 | 969 | 13.9 | 0.0 | 63 | 122/122 |
| run07_env8_gpu0_noprealloc_nullclient | 8 | GPU0 | lazy | null | 481 | 19 | 140/152 | 92.1 | 19.1 | 57.7 | 22.0 | 2.75 | 1174 | 16.9 | 0.0 | 70 | 140/140 |
| run07_env10_gpu0_noprealloc_nullclient | 10 | GPU0 | lazy | null | 481 | 17 | 151/170 | 88.8 | 16.5 | 53.9 | 26.5 | 2.65 | 1204 | 19.5 | 0.0 | 72 | 151/151 |
| run07_env12_gpu0_noprealloc_nullclient | 12 | GPU0 | lazy | null | 481 | 14 | 148/168 | 88.1 | 16.8 | 62.4 | 30.4 | 2.53 | 1228 | 22.8 | 0.0 | 75 | 148/148 |
