# 职业健康大数据智能决策系统

职业健康大数据智能决策系统初步构建，主要模块包含staff_info、factory_info、hazard_info、detection_info以及diagnose_info。

根据加载的职业健康大数据进行研究分析，主要支持两个方面：

1. 针对特定职业病的研究分析，如针对噪声性耳聋的分析；
2. 针对地区职业病的研究分析，如地区职业病爆发趋势研究；

## 目录结构

``` python
OCCUPATIONAL_HEALTH_SYSTEM_V2/
├── constants/
│   ├── auditory_constants.py
│   ├── global_constants.py
│   ├── industry_constants.py
│   └── ...
├── detection_info/
│   ├── base_result.py
│   ├── curve_result.py
│   ├── point_result.py
│   ├── auditory_detection/
│   |   ├── ABR_result.py
│   |   ├── PTA_result.py
│   │   └── ...
│   └── ...
├── diagnose_info/
│   ├── auditory_diagnose.py
│   └── ...
├── factory_info/
│   └── ...
├── hazard_info/
│   ├── base_hazard.py
│   ├── noise_hazard.py
│   └── ...
├── model/
│   ├── train_model.py
│   ├── fit_function/
│   |   ├── LAeq_functions.py
│   |   ├── NIL_functions.py
│   |   └── ...
│   ├── multi_task/
│   |   ├── esmm.py
│   |   ├── mmoe.py
│   |   ├── model_train.py
│   |   └── ...
│   ├── tab_transfromer/
│   |   ├── fit_transformer.py
│   |   ├── tab_transformer_pytorch.py
│   |   └── ...
│   └── ...
├── staff_info/
│   ├── staff_basic_info.py
│   ├── staff_health_info.py
│   ├── staff_info.py
│   ├── staff_occupational_hazard_info.py
│   └── ...
├── utils/
│   ├── data_helper.py
│   ├── database_helper.py
│   ├── decorators.py
│   ├── plot_helper.py
│   └── ...
├── examples/
├── .gitignore
├── requirements.txt
└── README.md
```

## 模块和组件

* `staff_info`: 用于将各类信息汇总至人的维度；
* `factory_info`: 用于工厂环境检测信息汇总至工厂维度；
* `hazard_info`: 用于整合各种不同类型的职业危害因素的检测信息；
* `detection_info`: 用于整合各种不同类型的体检项目的直接结果；
* `diganose_info`: 用于在体检检查信息的基础上进一步进行数据加工及诊断；
* `model`: 用于存放各类分析模型；
* `utils`: 用于存放常用工具；
* `examples`: 用于存放具体的研究分析实例；
