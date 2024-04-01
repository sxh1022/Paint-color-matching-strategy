# -- coding: utf-8 --
# Project ：color_center 
# File    ：lab_2_rgb.py
# Author  ：全俊洁
# Company ：Enoch
# Date    ：2023-05-17 16:33
# Desc    : 根据档级进行划分
from config import default
import numpy as np

def interval(a, start, end, inclusive='both'):
    if inclusive == 'both':
        return start <= a <= end
    elif inclusive == 'right':
        return start < a <= end
    elif inclusive == 'left':
        return start <= a < end

def calculate_score_with_weights(de_total, de_sg, de_dc, de_cv, de_h15, de_h45, de_h110):
    if np.isnan(de_total) or np.isnan(de_sg) or np.isnan(de_dc) or np.isnan(de_cv) or np.isnan(de_h15) or np.isnan(
            de_h45) or np.isnan(de_h110):
        return -1
    de_total, de_h15, de_h45, de_h110 = int(de_total * 10) / 10.0, abs(de_h15), abs(de_h45), abs(de_h110)
    de_sg, de_dc, de_cv = int(de_sg * 10) / 10.0, int(de_dc * 10) / 10.0, int(de_cv * 10) / 10.0,

    # 基于priority进行加权分设置
    add_value_json = {
        1: 3, 2: 2, 3: 1, 4: 0, 5: -1, 6: 0, 7: 0, 8: -1,
        9: 3, 10: 2, 11: 1, 12: 0, 13: -1, 14: 0, 15: -1
    }
    # 权重
    weights = default.weights
    thresholds, basic_score, max_score, weighted_deltas, priority_redefine = None, None, None, None, None

    # 评分机制设计
    """
    1.deltaE ∈ [0, 1.4]
        a.所有纹理阈值均在L1,L2内  得分范围在[90, 100)
        b.部分纹理阈值在L3，L4内   得分范围在[85, 95)
        c.部分纹理阈值在L5内       得分范围在[80, 90)
    2.deltaE ∈ (1.4, 3]
        a.所有纹理阈值均在L1,L2内  得分范围在[85, 95)
        b.部分纹理阈值在L3，L4内   得分范围在[75, 85)
        c.部分纹理阈值在L5内       得分范围在[65, 75)
    """
    if de_total <= default.Metal_De:
        # a
        if de_sg <= default.Sg_Threshold and de_dc <= default.Dc_Threshold and de_cv <= default.Cv_Threshold:
            thresholds = [default.Metal_De, default.Dc_Threshold, default.Sg_Threshold,
                          default.Cv_Threshold] + default.deltaH_threshold
            basic_score = 90
            max_score = 100
            # 计算每个变量与对应阈值的加权值
            weighted_deltas = [w * (t - x) / t for x, t, w in
                               zip([de_total, de_dc, de_sg, de_cv, de_h15, de_h45, de_h110], thresholds, weights)]
        # b
        elif de_sg <= default.Sg_Threshold_Second and de_dc <= default.Dc_Threshold_Second and de_cv <= default.Cv_Threshold_Second:
            # 参数变化范围
            thresholds_delta = [1.4, default.Dc_Threshold_Second - default.Dc_Threshold,
                                default.Sg_Threshold_Second - default.Sg_Threshold,
                                default.Cv_Threshold_Second - default.Cv_Threshold] + default.deltaH_threshold
            thresholds = [default.Metal_De, default.Dc_Threshold_Second, default.Sg_Threshold_Second,
                          default.Cv_Threshold_Second] + default.deltaH_threshold
            basic_score = 85
            max_score = 96
            # 计算每个变量与对应阈值的加权值
            weighted_deltas = [w * (t - x) / d for x, t, w, d in
                               zip([de_total, de_dc, de_sg, de_cv, de_h15, de_h45, de_h110], thresholds, weights,
                                   thresholds_delta)]
        # c
        elif de_sg <= default.Sg_Threshold_Third and de_dc <= default.Dc_Threshold_Third and de_cv <= default.Cv_Threshold_Third:
            # 参数变化范围
            thresholds_delta = [1.4, default.Dc_Threshold_Third - default.Dc_Threshold_Second,
                                default.Sg_Threshold_Third - default.Sg_Threshold_Second,
                                default.Cv_Threshold_Third - default.Cv_Threshold_Second] + default.deltaH_threshold
            thresholds = [default.Metal_De, default.Dc_Threshold_Third, default.Sg_Threshold_Third,
                          default.Cv_Threshold_Third] + default.deltaH_threshold
            basic_score = 80
            max_score = 90
            # 计算每个变量与对应阈值的加权值
            weighted_deltas = [w * (t - x) / d for x, t, w, d in
                               zip([de_total, de_dc, de_sg, de_cv, de_h15, de_h45, de_h110], thresholds, weights,
                                   thresholds_delta)]
    else:
        weights = default.weights_second
        if de_sg <= default.Sg_Threshold and de_dc <= default.Dc_Threshold and de_cv <= default.Cv_Threshold:
            thresholds = [default.Metal_De_Threshold_Second, default.Dc_Threshold, default.Sg_Threshold,
                          default.Cv_Threshold] + default.deltaH_threshold
            thresholds_delta = [1.6, default.Dc_Threshold, default.Sg_Threshold,
                                default.Cv_Threshold] + default.deltaH_threshold
            basic_score = 85
            max_score = 95
            # 计算每个变量与对应阈值的加权值
            weighted_deltas = [w * (t - x) / d for x, t, w, d in
                               zip([de_total, de_dc, de_sg, de_cv, de_h15, de_h45, de_h110], thresholds, weights,
                                   thresholds_delta)]
        # b
        elif de_sg <= default.Sg_Threshold_Second and de_dc <= default.Dc_Threshold_Second and de_cv <= default.Cv_Threshold_Second:
            # 参数变化范围
            thresholds = [default.Metal_De_Threshold_Second, default.Dc_Threshold_Second, default.Sg_Threshold_Second,
                          default.Cv_Threshold_Second] + default.deltaH_threshold
            thresholds_delta = [1.6, default.Dc_Threshold_Second - default.Dc_Threshold,
                                default.Sg_Threshold_Second - default.Sg_Threshold,
                                default.Cv_Threshold_Second - default.Cv_Threshold] + default.deltaH_threshold
            basic_score = 75
            max_score = 85
            # 计算每个变量与对应阈值的加权值
            weighted_deltas = [w * (t - x) / d for x, t, w, d in
                               zip([de_total, de_dc, de_sg, de_cv, de_h15, de_h45, de_h110], thresholds, weights,
                                   thresholds_delta)]
        # c
        elif de_sg <= default.Sg_Threshold_Third and de_dc <= default.Dc_Threshold_Third and de_cv <= default.Cv_Threshold_Third:
            # 参数变化范围
            thresholds_delta = [1.6, default.Dc_Threshold_Third - default.Dc_Threshold_Second,
                                default.Sg_Threshold_Third - default.Sg_Threshold_Second,
                                default.Cv_Threshold_Third - default.Cv_Threshold_Second] + default.deltaH_threshold
            thresholds = [default.Metal_De, default.Dc_Threshold_Third, default.Sg_Threshold_Third,
                          default.Cv_Threshold_Third] + default.deltaH_threshold
            basic_score = 65
            max_score = 85
            # 计算每个变量与对应阈值的加权值
            weighted_deltas = [w * (t - x) / d for x, t, w, d in
                               zip([de_total, de_dc, de_sg, de_cv, de_h15, de_h45, de_h110], thresholds, weights,
                                   thresholds_delta)]
    if weighted_deltas is None:
        return -1
    total_weighted_delta = sum(weighted_deltas)
    # 计算综合评分值
    score = basic_score + (max_score - basic_score) * (total_weighted_delta / sum(weights))
    score = max(basic_score, round(min(max_score - 1, score), 1))
    return score

# # 93.6
# result = calculate_score_with_weights(1.57205121348, 3.8722261867920342, 4.551521445424895, 16.908370624098605,
#                                       24.39134308578062, 10.402804215212313, 11.441391203114264)
# print(result)
