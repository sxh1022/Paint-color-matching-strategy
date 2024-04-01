# -- coding: utf-8 --
# Project ：color_center 
# File    ：lab_2_rgb.py
# Author  ：全俊洁
# Company ：Enoch
# Date    ：2023-05-17 16:33
# Desc    : ${}
from config import default
import numpy as np


def interval(a, start, end, inclusive='both'):
    if inclusive == 'both':
        return start <= a <= end
    elif inclusive == 'right':
        return start < a <= end
    elif inclusive == 'left':
        return start <= a < end

def calculate_score_with_weights(de_total, vehicle_brand_match, vehicle_brand_input, vehicle_spec_match,
                                 vehicle_spec_input, de_h15, de_h45, de_h110):
    if np.isnan(de_total) or np.isnan(de_h15) or np.isnan(de_h45) or np.isnan(de_h110):
        return -1
    de_total = int(de_total * 10) / 10.0
    de_h15, de_h45, de_h110 = abs(de_h15), abs(de_h45), abs(de_h110)

    add_value_json = {
        1: 1, 4: 0, 6: -1
    }

    # 阈值
    thresholds = [default.De_Max] + default.deltaH_threshold
    weights = default.solid_weights
    basic_score = 70
    max_score = 100
    # 计算每个变量与对应阈值的加权值
    weighted_deltas = [w * (t - x) / t for x, t, w in
                       zip([de_total, de_h15, de_h45, de_h110], thresholds, weights)]

    total_weighted_delta = sum(weighted_deltas)
    # 计算综合评分值
    score = basic_score + (max_score - basic_score) * (total_weighted_delta / sum(weights))
    score = max(basic_score, round(min(max_score - 1, score), 1))

    brand_score = min(99, score + 2 if vehicle_brand_match == vehicle_brand_input else score)
    spec_score = min(99, brand_score + 3 if vehicle_spec_match == vehicle_spec_input else brand_score)
    return spec_score

# print(calculate_score_with_weights(1, '', None, '', None, 20, 3, 5))
