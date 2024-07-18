# 噪声保护临界值讨论

复现NOISH, 1998关于噪声保护临界值的论文内容，并在中国工人数据上进行相关研究。

关键程序及文件说明：

* Chinese_logistic_regresssion.py: 中国工人数据实验的最终结果，详细结果与讨论参见"D:\工作文档\研究课题\0.噪声与耳毒性化学溶剂联合暴露对听力损伤的影响\2.研究结论\复杂噪声影响\3.final_report\噪声性耳聋概率曲线拟合结果_0405.docx"；
* load_Chinese_data.py: 加载目前所有的中国工人噪声暴露数据（含C计权），构建staff_info对象后封存到pkl文件中；
* extract_all_Chinese_data.py: 从压缩的staff_info对象文件中进行信息提取；
* Chinese_control_group_logistic_regression_0.py: 中国工人对照组（含70dB以下暴露数据）数据听力损伤概率，自变量仅为age；
* Chinese_control_group_logistic_regression_1.py: 中国工人对照组（含70dB以下暴露数据）数据听力损伤概率，自变量为age与duration；
* Chinese_all_data_statistic_plot.py: 对所有中国工人数据（实验组+对照组）进行整体统计与检验并绘图；

* ./cache/extract_Chinese_data.pkl: 加载的所有中国工人暴露数据压缩包；
* ./cache/extract_Chinese_control_data.pkl: 加载的所有中国对照组数据压缩包；
* ./cache/Chinese_extract_experiment_classifier_df.csv: 最终结果中提取的中国工人暴露分类数据（以后其他分类过程中也应使用同一份文件）；
* ./cache/Chinese_extract_control_classifier_df.csv: 提取的中国工人对照组（含70dB以下暴露数据）分类数据；
* ./models/*: 目前训练好的有效模型；

## 2024.07.17

重新针对NOISH数据进行复现，详细对照数值结果是否与原文一致。需要探究background risk的数值获取。
