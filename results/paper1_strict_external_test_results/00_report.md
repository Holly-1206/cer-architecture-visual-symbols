# Strict external test and taxonomy stress test

- historical: 1044 images, 389 outside the taxonomy (376 with the explicit note)
- field: 1187 images, 0 outside the taxonomy (0 with the explicit note)

## External performance, all vs strict (cluster bootstrap, 2,000 replicates)

| domain     | subset   |   n_images |   n_clusters | method            |   macro_AUROC |   AUROC_low |   AUROC_high |   mAP |   mAP_low |   mAP_high |   macro_AUROC_lineages_n_ge_15 |   n_lineages_n_ge_15 |   macro_F1 |
|:-----------|:---------|-----------:|-------------:|:------------------|--------------:|------------:|-------------:|------:|----------:|-----------:|-------------------------------:|---------------------:|-----------:|
| historical | all      |       1044 |          902 | ConvNeXt-Tiny     |         0.618 |       0.577 |        0.662 | 0.174 |     0.155 |      0.21  |                          0.623 |                    8 |      0.093 |
| historical | all      |       1044 |          902 | DINOv2 prototypes |         0.73  |       0.682 |        0.768 | 0.311 |     0.271 |      0.377 |                          0.665 |                    8 |    nan     |
| historical | strict   |        655 |          610 | ConvNeXt-Tiny     |         0.594 |       0.546 |        0.642 | 0.168 |     0.147 |      0.211 |                          0.583 |                    7 |      0.069 |
| historical | strict   |        655 |          610 | DINOv2 prototypes |         0.701 |       0.642 |        0.751 | 0.311 |     0.264 |      0.38  |                          0.626 |                    7 |    nan     |
| field      | all      |       1187 |          258 | ConvNeXt-Tiny     |         0.617 |       0.539 |        0.656 | 0.191 |     0.144 |      0.244 |                          0.596 |                   11 |      0.145 |
| field      | all      |       1187 |          258 | DINOv2 prototypes |         0.663 |       0.608 |        0.687 | 0.245 |     0.187 |      0.305 |                          0.649 |                   11 |    nan     |
| field      | strict   |       1187 |          258 | ConvNeXt-Tiny     |         0.617 |       0.54  |        0.656 | 0.191 |     0.145 |      0.242 |                          0.596 |                   11 |      0.145 |
| field      | strict   |       1187 |          258 | DINOv2 prototypes |         0.663 |       0.609 |        0.689 | 0.245 |     0.188 |      0.307 |                          0.649 |                   11 |    nan     |

## Prototypes minus ConvNeXt

| domain     | subset   |   AUROC_difference |   AUROC_diff_low |   AUROC_diff_high |   mAP_difference |   mAP_diff_low |   mAP_diff_high |
|:-----------|:---------|-------------------:|-----------------:|------------------:|-----------------:|---------------:|----------------:|
| historical | all      |              0.112 |            0.043 |             0.168 |            0.137 |          0.093 |           0.187 |
| historical | strict   |              0.107 |            0.027 |             0.175 |            0.143 |          0.1   |           0.197 |
| field      | all      |              0.046 |            0.009 |             0.099 |            0.053 |          0.027 |           0.088 |
| field      | strict   |              0.046 |            0.008 |             0.096 |            0.053 |          0.024 |           0.087 |

## Taxonomy stress test: 389 album images outside the twelve lineages

| lineage                     |   annotators_assigned |   prototype_nearest |   classifier_top |
|:----------------------------|----------------------:|--------------------:|-----------------:|
| 01 Russian Traditional      |                     0 |                   2 |                8 |
| 02 Russian Timber Dwelling  |                     0 |                  11 |                0 |
| 03 Russian Railway          |                    25 |                   5 |               97 |
| 04 Byzantine / Orthodox     |                    40 |                   1 |               20 |
| 05 Eclecticism              |                    39 |                   0 |               12 |
| 06 Art Nouveau              |                     0 |                   5 |               74 |
| 07 Baroque                  |                     0 |                   4 |               29 |
| 08 Chinese Baroque          |                   290 |                 351 |              110 |
| 09 Classical / Neoclassical |                     0 |                   0 |                5 |
| 10 Gothic                   |                     0 |                   2 |               12 |
| 11 Art Deco                 |                     0 |                   6 |               12 |
| 12 Early Modernism          |                     0 |                   2 |               10 |

Share nearest the Chinese Baroque prototype: 351/389 (90%); classifier top = Chinese Baroque: 110/389 (28%). In-taxonomy album images nearest Chinese Baroque: 122/655 (19%).

## Images flagged by building type but without the explicit note

| stem                                                                                                        |   style_ids |
|:------------------------------------------------------------------------------------------------------------|------------:|
| D1107_辽阳（沿线其他城市）_室内与器物陈设_辽阳天齐庙神像二__中国传统宗教建筑_天齐庙木板内墙背景             |          08 |
| D1078_沈阳（沿线其他城市）_建筑构件与装饰_沈阳寺庙内部空间彩绘__中国传统宗教建筑内部装饰_梁枋斗拱与藻井彩绘 |          08 |
| D483_营口（沿线其他城市）_行政办公与公共建筑_营口河岸公共建筑(1)__中国传统建筑_表外补充_折衷主义            |          08 |
| D1105_辽阳（沿线其他城市）_室内与器物陈设_辽阳娘娘庙神像二__中国传统宗教建筑_娘娘庙神龛卷云彩画背板         |          08 |
| D1072_沈阳（沿线其他城市）_建筑构件与装饰_沈阳寺庙壁画__中国传统宗教建筑内部装饰_人物故事壁画与木框         |          08 |
| D1104_辽阳（沿线其他城市）_室内与器物陈设_辽阳娘娘庙神像一__中国传统宗教建筑_娘娘庙神龛木柱栏杆与砖石台基   |          08 |
| D482_营口（沿线其他城市）_宗教建筑_营口寺庙与街市(1)__中国传统建筑_表外补充                                 |          08 |
| D1071_沈阳（沿线其他城市）_建筑构件与装饰_沈阳寺庙内部透视彩绘(1)__中国传统宗教建筑_室内透视彩画与木构框架  |          08 |
| D1080_沈阳（沿线其他城市）_建筑构件与装饰_沈阳寺庙山水壁画__中国传统宗教建筑内部装饰_山水壁画与雕花木框     |          08 |
| D1106_辽阳（沿线其他城市）_室内与器物陈设_辽阳天齐庙神像一__中国传统宗教建筑_天齐庙神龛木柱砖墙与彩画背板   |          08 |
| D1108_海城（沿线其他城市）_室内与器物陈设_海城上帝庙神像__中国传统宗教建筑_上帝庙神龛木框画屏与砖石墙裙     |          08 |
| D481_营口（沿线其他城市）_城市道路与公共空间_营口街市与商铺(1)__中国传统建筑_表外补充                       |          08 |
| D1103_辽阳（沿线其他城市）_室内与器物陈设_辽阳近郊龙王庙神像(2)__中国传统宗教建筑_龙王庙神龛画屏与木构边框  |          08 |