# Final Experiment Results

- Generated at: `2026-06-23T06:01:54`
- Dataset stage: `processed`
- Fixed YAML-driven run order; no dynamic branching.

## Run Order

1. `e0-baseline-medium`
2. `e0-baseline-nano`
3. `e0-baseline-large`
4. `e1-dkn`
5. `e1-ibv`
6. `e1-public-noaug`
7. `e2-public-default`
8. `e2-public-busaug`
9. `e4-private-adapt-medium`
10. `e4-private-adapt-medium-busaug`
11. `e5-private-adapt-nano`
12. `e5-private-adapt-large`
13. `e5-private-adapt-nano-busaug`
14. `e5-private-adapt-large-busaug`
15. `e3-public-crowd-default`
16. `e3-public-crowd-busaug`

## Per-dataset Metrics

| experiment | dataset | mAP50 | F1 | count_mae | count_me | count_rmse |
|---|---:|---:|---:|---:|---:|---:|
| e0-baseline-medium | passenger-detection-bus | 0.0635 | 0.1504 | 1.7059 | -0.8824 | 2.1282 |
| e0-baseline-medium | inside-bus-view | 0.3286 | 0.3739 | 9.2667 | -8.8667 | 11.7104 |
| e0-baseline-medium | passenger-deakin | 0.2141 | 0.3168 | 2.5773 | -1.0928 | 3.8423 |
| e0-baseline-medium | onibus-unicamp-private | 0.6117 | 0.6119 | 1.1847 | -1.0318 | 1.9220 |
| e0-baseline-nano | passenger-detection-bus | 0.0723 | 0.1535 | 3.2941 | -3.0588 | 4.9229 |
| e0-baseline-nano | inside-bus-view | 0.2404 | 0.2621 | 11.8667 | -11.8667 | 14.1939 |
| e0-baseline-nano | passenger-deakin | 0.1724 | 0.2703 | 3.2784 | -1.6495 | 4.5657 |
| e0-baseline-nano | onibus-unicamp-private | 0.5694 | 0.5861 | 1.4331 | -1.3949 | 2.2016 |
| e0-baseline-large | passenger-detection-bus | 0.0669 | 0.1667 | 2.1176 | -0.9412 | 2.8284 |
| e0-baseline-large | inside-bus-view | 0.2969 | 0.3398 | 10.6667 | -10.5333 | 13.1859 |
| e0-baseline-large | passenger-deakin | 0.2144 | 0.3271 | 2.6598 | -1.2165 | 4.0334 |
| e0-baseline-large | onibus-unicamp-private | 0.6318 | 0.6243 | 1.1146 | -0.9745 | 1.8182 |
| e1-dkn | passenger-detection-bus | 0.4740 | 0.5168 | 3.9412 | -3.9412 | 5.5147 |
| e1-dkn | inside-bus-view | 0.4404 | 0.5404 | 5.5333 | -5.5333 | 7.0852 |
| e1-dkn | passenger-deakin | 0.6504 | 0.6786 | 2.0928 | 0.3814 | 3.2220 |
| e1-dkn | onibus-unicamp-private | 0.1258 | 0.2400 | 1.8153 | -1.7261 | 2.4741 |
| e1-ibv | passenger-detection-bus | 0.6801 | 0.6284 | 79.1765 | 79.1765 | 80.7596 |
| e1-ibv | inside-bus-view | 0.6328 | 0.6169 | 44.8000 | 44.8000 | 45.7690 |
| e1-ibv | passenger-deakin | 0.1895 | 0.2844 | 98.6598 | 98.6598 | 107.4492 |
| e1-ibv | onibus-unicamp-private | 0.0116 | 0.0424 | 52.3312 | 52.3312 | 55.4109 |
| e1-public-noaug | passenger-detection-bus | 0.4841 | 0.5532 | 5.4118 | -5.1765 | 6.1261 |
| e1-public-noaug | inside-bus-view | 0.7565 | 0.7685 | 3.2000 | -3.2000 | 4.1473 |
| e1-public-noaug | passenger-deakin | 0.5425 | 0.5722 | 2.9691 | -2.0619 | 4.2280 |
| e1-public-noaug | onibus-unicamp-private | 0.0699 | 0.1571 | 2.3694 | -2.3439 | 3.0848 |
| e2-public-default | passenger-detection-bus | 0.5686 | 0.6235 | 3.5882 | -3.1176 | 4.4125 |
| e2-public-default | inside-bus-view | 0.8521 | 0.8410 | 0.7333 | -0.0667 | 0.9309 |
| e2-public-default | passenger-deakin | 0.6660 | 0.6643 | 1.8454 | 0.0515 | 2.8339 |
| e2-public-default | onibus-unicamp-private | 0.1341 | 0.2216 | 1.9554 | -1.9045 | 2.7681 |
| e2-public-busaug | passenger-detection-bus | 0.3073 | 0.4431 | 2.8235 | -2.7059 | 3.6782 |
| e2-public-busaug | inside-bus-view | 0.8685 | 0.8444 | 1.6667 | 1.1333 | 2.2657 |
| e2-public-busaug | passenger-deakin | 0.6445 | 0.6467 | 2.3918 | 1.1959 | 3.4849 |
| e2-public-busaug | onibus-unicamp-private | 0.1751 | 0.2805 | 1.6369 | -1.3949 | 2.3391 |
| e4-private-adapt-medium | passenger-detection-bus | 0.5841 | 0.6429 | 2.2941 | -1.7059 | 3.2988 |
| e4-private-adapt-medium | inside-bus-view | 0.8579 | 0.8442 | 1.6000 | -1.2000 | 2.1909 |
| e4-private-adapt-medium | passenger-deakin | 0.6592 | 0.6700 | 1.8660 | 0.5258 | 2.9022 |
| e4-private-adapt-medium | onibus-unicamp-private | 0.7888 | 0.7649 | 0.6815 | -0.0064 | 1.1702 |
| e4-private-adapt-medium-busaug | passenger-detection-bus | 0.5244 | 0.5751 | 3.2353 | -2.0588 | 4.0511 |
| e4-private-adapt-medium-busaug | inside-bus-view | 0.8595 | 0.8444 | 1.0000 | 0.4667 | 1.2910 |
| e4-private-adapt-medium-busaug | passenger-deakin | 0.6549 | 0.6595 | 1.9897 | 0.6495 | 3.1180 |
| e4-private-adapt-medium-busaug | onibus-unicamp-private | 0.7848 | 0.7510 | 0.7516 | 0.1401 | 1.1455 |
| e5-private-adapt-nano | passenger-detection-bus | 0.4636 | 0.5328 | 2.4706 | -0.4706 | 3.2358 |
| e5-private-adapt-nano | inside-bus-view | 0.8608 | 0.8379 | 1.6000 | 1.3333 | 2.3944 |
| e5-private-adapt-nano | passenger-deakin | 0.6869 | 0.6883 | 1.9485 | 0.8557 | 3.1508 |
| e5-private-adapt-nano | onibus-unicamp-private | 0.7845 | 0.7475 | 0.8599 | 0.3758 | 1.2992 |
| e5-private-adapt-large | passenger-detection-bus | 0.4663 | 0.5634 | 2.0588 | -1.7059 | 2.8800 |
| e5-private-adapt-large | inside-bus-view | 0.8212 | 0.8099 | 1.9333 | -0.7333 | 2.4083 |
| e5-private-adapt-large | passenger-deakin | 0.6553 | 0.6433 | 2.0000 | 0.0825 | 2.9672 |
| e5-private-adapt-large | onibus-unicamp-private | 0.7632 | 0.7059 | 0.7898 | 0.2166 | 1.3355 |
| e5-private-adapt-nano-busaug | passenger-detection-bus | 0.3284 | 0.4377 | 2.2941 | -1.5882 | 2.8180 |
| e5-private-adapt-nano-busaug | inside-bus-view | 0.8546 | 0.8362 | 1.2667 | -0.3333 | 1.5706 |
| e5-private-adapt-nano-busaug | passenger-deakin | 0.6400 | 0.6483 | 2.0000 | 0.7216 | 2.9741 |
| e5-private-adapt-nano-busaug | onibus-unicamp-private | 0.7999 | 0.7620 | 0.7452 | 0.0955 | 1.2390 |
| e5-private-adapt-large-busaug | passenger-detection-bus | 0.5565 | 0.6336 | 2.7647 | -2.5294 | 3.5892 |
| e5-private-adapt-large-busaug | inside-bus-view | 0.8375 | 0.8338 | 2.0000 | -0.6667 | 2.6833 |
| e5-private-adapt-large-busaug | passenger-deakin | 0.6470 | 0.6407 | 2.2784 | 0.7113 | 3.5624 |
| e5-private-adapt-large-busaug | onibus-unicamp-private | 0.7584 | 0.7343 | 0.8025 | 0.1529 | 1.2869 |
| e3-public-crowd-default | passenger-detection-bus | 0.6051 | 0.6323 | 2.2941 | -1.8235 | 3.3694 |
| e3-public-crowd-default | inside-bus-view | 0.8603 | 0.8305 | 1.7333 | 0.4000 | 2.0331 |
| e3-public-crowd-default | passenger-deakin | 0.6316 | 0.6288 | 2.4536 | 1.1546 | 3.3980 |
| e3-public-crowd-default | onibus-unicamp-private | 0.1561 | 0.2532 | 1.8025 | -1.6624 | 2.5898 |
| e3-public-crowd-busaug | passenger-detection-bus | 0.3581 | 0.4411 | 3.8235 | -3.7059 | 4.8203 |
| e3-public-crowd-busaug | inside-bus-view | 0.8471 | 0.8323 | 0.8667 | 0.0667 | 1.1832 |
| e3-public-crowd-busaug | passenger-deakin | 0.6633 | 0.6639 | 1.9381 | 0.4536 | 2.9983 |
| e3-public-crowd-busaug | onibus-unicamp-private | 0.1852 | 0.2962 | 1.8535 | -1.7771 | 2.6313 |

