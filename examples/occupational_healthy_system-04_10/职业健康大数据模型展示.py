# -*- coding: utf-8 -*-
"""
@DATE: 2024-04-10 09:42:00
@Author: Liu Hengjiang
@File: examples\occupational_healthy_system-04_10\streamlit_show.py
@Software: vscode
@Description:
        整合所有功能统一进行展示
"""

import streamlit as st

if __name__ == "__main__":
    st.write("# 职业健康大数据模型案例展示")
    st.sidebar.success("点击标题查看具体案例")

    st.markdown("""
        近年来，随着互联网以及大数据技术的快速发展，针对职业健康的大数据的挖掘与应用逐渐受到研究者们的关注。
        
        据统计，至2022年底，我国的劳动年龄人口近9亿，丰富的劳动力资源为职业病防治工作积累了海量的数据。
        
        如果能有效地组织、开发和利用职业病防治大数据，将对职业病检测预警、风险评估、应急处理、诊断治疗以
        及政策制定等职业健康管理工作产生巨大的推动作用。
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("## 职业健康大数据的来源")
        st.markdown("""
                    * 政府及职业病防治技术机构储存数据；
                    * 以人群为基础的医疗和社会化活动信息；
                    * 以企业为基础的工厂环境及职业病预防控制信息；
                    * 网络多媒体舆情数据；
        """)
    with col2:
        st.markdown("## 以数据为主体的应用领域")
        st.markdown("""
                    * 职业病预防控制政策制定和资源优化；
                    * 职业病监测和风险评估；
                    * 量化的职业病疾病负担、 工伤保险计算 ；
                    * 职业病舆情检测和预警；
        """)

    st.markdown("## 职业健康大数据应用构建")
    st.markdown("### 数据采集")
    st.image("/mount/src/occupational_health_system_v2/examples/occupational_healthy_system-04_10/pictures/系统架构图-彩-Part1.png")
    # st.image("./pictures/系统架构图-彩-Part1.png")
    st.markdown("""
                * 工作场所环境数据库：所属行业类别、地址、经纬度、海拔、工作状态情况（如工作时长、流水线状况、是否配备有保护装置等），工作环境中的空间大小、人员密度、空气质量（粉尘、化学物质浓度等）、噪声、光照、辐射情况以及其他可能的风险因素；
                * 个人医疗信息数据库： 基本健康信息（年龄、性别、职业、婚姻状况、工作地点、工作岗位等）、卫生事件摘要信息（就诊记录、就诊原因、药物记录等）、医疗费用记录信息（诊疗费用记录、保险记录等）；
                * 实验数据：从部分工作人员所佩戴的随身检测设备上获取到一些实时、动态的检测数据；
    """)
    st.markdown("### 职业健康信息管理")
    st.image("/mount/src/occupational_health_system_v2/examples/occupational_healthy_system-04_10/pictures/系统架构图-彩-Part2.png")
    # st.image("./pictures/系统架构图-彩-Part2.png")
    st.markdown("""
                * 按照国家卫健委有关职业病分类和目录的相关信息，以及职业病的病理学相关知识进行领域建模，完成数据的初步归类与标注，按数据源、数据集成、数据集市的数仓结构进行信息管理；
                * 围绕多层级的库表结构进行数据的脱敏加密与权限管理，消除敏感信息。隔离下游应用中直接访问敏感数据的途径，保护数据的隐私安全；
    """)
    st.markdown("### 数据处理与模型分析")
    st.image("/mount/src/occupational_health_system_v2/examples/occupational_healthy_system-04_10/pictures/系统架构图-彩-Part3.png")
    # st.image("./pictures/系统架构图-彩-Part3.png")
    st.markdown("""
                * 服务对象：研究人员；
                * 主要功能：对无效数据的清洗与去噪、数据采样以及针对不同职业病数据的模型分析；
                * 拓展功能：以模块化的方式接入更多分析功能模块，如地区职业健康风险监控、职工职业疾患病风险监控等；
    """)
    st.markdown("### 信息应用与共享")
    st.image("/mount/src/occupational_health_system_v2/examples/occupational_healthy_system-04_10/pictures/系统架构图-彩-Part4.png")
    # st.image("./pictures/系统架构图-彩-Part4.png")
    st.markdown("""
                * 服务对象：政府及监管机构、企业、个体工作人员、研究人员
                * 主要功能：数据可视化、报告自动生成、提供结构化数据
    """)
    