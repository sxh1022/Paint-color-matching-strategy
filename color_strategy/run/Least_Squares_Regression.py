# coding=utf-8
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from color_strategy.Utils import run_data, sql_data
import itertools
from color_strategy.Utils.utils import delta_e, delta_e_3angle, Refl_2_Lab, delta_e_3angle_list, delta_wl
from color_strategy.Utils import metal_level_score
from color_strategy.Utils import solid_level_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
import colour
import math
import uvicorn
from fastapi import FastAPI
import logging
import ast
from fastapi.responses import JSONResponse
from scipy.stats import entropy
from color_strategy.Utils.run_data import *
from scipy.optimize import minimize
import time
from color_strategy import Utils
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

def get_formula_type(formula_code, particle_colorants):
    """
    返回配方中颗粒色母的数量
    """
    non_zero_columns = [col for col in particle_colorants if formula_code[col].any()]
    return len(non_zero_columns)


def model_get_important_feature(df, features, targets):
    """
    使用随机森林回归模型获取特征重要性，排除所有值为0的特征
    """
    non_zero_features = [feature for feature in features if not df[feature].eq(0).all()]

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(df[non_zero_features], df[targets])

    result = permutation_importance(model, df[non_zero_features], df[targets], n_repeats=5, random_state=0)

    feature_importances = sorted(zip(result.importances_mean, non_zero_features), reverse=True)
    sorted_features = [feature for importance, feature in feature_importances]

    excluded_features = [feature for feature in features if feature not in non_zero_features]
    sorted_features += excluded_features

    return np.array(sorted_features)


def mutual_info_get_important_feature(df, features, targets):
    """
    互信息获取特征重要性
    """
    feature_rankings = [
        pd.DataFrame({'Feature': features,
                      'Mutual Information': mutual_info_regression(df[features], df[target]),
                      'Target': target})
        for target in targets]

    feature_importances = pd.concat(feature_rankings).groupby('Feature')['Mutual Information'].apply(
        lambda x: abs(x).mean()).sort_values(ascending=False)

    sorted_features = feature_importances.index
    return np.array(sorted_features)


def sort_colorants_result(df, features, targets, match_colorants, order_type="forest"):
    # start_time = time.time()
    """
    返回最终的色母重要性排序
    :param order_type: 可选mutual_info和forest，默认为随机森林
    """
    if order_type == 'mutual_info':
        feature_list = mutual_info_get_important_feature(df, features, targets)
    else:
        feature_list = model_get_important_feature(df, features, targets)
    # 将基础配方中的色母提前
    final_feature_order = [feature for feature in feature_list if feature in match_colorants] + \
                          [feature for feature in feature_list if feature not in match_colorants]
    final_feature_order = [feature for feature in feature_list]
    # end_time = time.time()
    # print("色母重要性排序耗时：", end_time - start_time)
    return final_feature_order


def find_best_subset_order_method(match_data_df, delta_colorants, formula, feature_list, match_colorants_list,
                                  match, wl, target_wl, match_wl, lab, target_lab, match_lab, origin_score,
                                  ignore_colorants, weights, flag=True):
    """
    寻找最佳的微调色母输出，按照顺序
    :param match_data_df: 匹配数据集
    :param delta_colorants: 微调策略计算出的基础配方的调整量
    :param formula: 全部色母列表
    :param feature_list: 色母排序
    :param match_colorants_list: 基础配方中的色母组成
    :param match: 基础配方
    :param wl: 纹理属性
    :param target_wl: 目标纹理信息
    :param match_wl: 基础配方纹理信息
    :param lab: 颜色属性
    :param target_lab: 目标颜色信息
    :param match_lab: 基础配方颜色信息
    :param origin_score: 基础配方的得分
    :param ignore_colorants: 忽略色母的列表
    :param weights: 权重
    :param flag: true代表颗粒配方，false代表素色配方
    :return: 最终的微调色母列表，以及是否可调
    """
    data_dict = dict(zip(formula, delta_colorants))
    all_df_delta_colorants = pd.DataFrame(data_dict, index=[0])

    best_subset = []
    best_sign_match = False
    scores = []

    # start_time = time.time()
    # 创建岭回归模型
    match_data_df[formula] = match_data_df[formula].astype(float)
    if flag:
        ridge_model = Ridge(alpha=0.1)
        ridge_model.fit(match_data_df[formula].values, match_data_df[lab + wl].values)

        train_accuracy = ridge_model.score(match_data_df[formula].values, match_data_df[lab + wl].values)
        # print("分析时ridge_model训练集的准确率: ", train_accuracy)
    else:
        ridge_model = Ridge(alpha=0.1)
        ridge_model.fit(match_data_df[formula].values, match_data_df[lab].values)

        train_accuracy = ridge_model.score(match_data_df[formula].values, match_data_df[lab].values)
        # print("分析时ridge_model训练集的准确率: ", train_accuracy)
    # end_time = time.time()
    # print("岭回归模型耗时：", end_time - start_time)
    for feature in feature_list:
        if feature not in ignore_colorants:
            feature_delta = all_df_delta_colorants.loc[0, feature]
            # if feature_delta != 0 and 0 <= match[feature] + feature_delta <= 100:
            if feature_delta != 0:
                temp_subset = best_subset + [feature]

                # 数据处理
                df_delta_colorants = pd.DataFrame(data_dict, index=[0])
                cols_to_zero = [col for col in df_delta_colorants.columns if col not in temp_subset]
                df_delta_colorants[cols_to_zero] = 0
                df_delta_colorants[formula] = df_delta_colorants[formula] + match[formula]
                df_delta_colorants = df_delta_colorants.astype(float)

                if flag:
                    temp_sign_match, new_del_total, new_dc, new_sg, new_cv, score, new_score, delta_sum = \
                        analysis_subset_deltaE_wl_trend(
                            ridge_model, df_delta_colorants, target_wl, match_wl, target_lab, match_lab, weights,
                            match['flag_code'])
                    # print(
                    #     "Colorant: {}, DeltaE_3: {}, Change: {}, DeltaE: {}, DeltaDC: {}, DeltaSG: {}, "
                    #     "DeltaCV: {}, score: {}, new_score: {}".format(
                    #         feature, temp_sign_match, feature_delta, new_del_total, new_dc, new_sg, new_cv, score,
                    #         new_score
                    #     ))
                    scores.append((score, new_score))
                else:
                    temp_sign_match, new_del_total, score, new_score = \
                        analysis_subset_deltaE_trend(ridge_model, df_delta_colorants, target_lab, match_lab, weights,
                                                     match['flag_code'])
                    # print(
                    #     "Colorant: {}, DeltaE_3: {}, Change: {},  DeltaE: {}, score: {}, new_score: {}".format(
                    #         feature, temp_sign_match, feature_delta, new_del_total, score, new_score
                    #     ))
                    scores.append((score, new_score))
                best_subset = temp_subset

    if len(scores) == 0:
        return [], False
    final_score, final_diff_score, count = optimal_score(scores, origin_score)
    # print("count: {}, score: {}".format(count, final_score))
    if final_score > origin_score:
        best_sign_match = True

    return best_subset[:count], best_sign_match


def analysis_subset_deltaE_wl_trend(model, df_delta_colorants, target_wl, match_wl, target_lab, match_lab, weights, flag_code):
    """
    针对颗粒配方，分析改变逐个色母时的色差和纹理差的变化情况
    :param model: 用来分析当前配方色差和纹理差的模型
    :param df_delta_colorants: 当前配方
    :param target_wl: 目标的纹理信息
    :param match_wl: 基础配方的纹理信息
    :param target_lab: 目标的颜色信息
    :param match_lab: 基础配方的颜色信息
    :param weights: 权重
    :return: 3角度ΔE是否都减小，新的色差，纹理差，两种分数，以及色差和纹理差的和
    """

    subset_delta_combined = model.predict(df_delta_colorants.values)
    subset_delta_lab = subset_delta_combined.flatten()[:9]
    subset_delta_wl = subset_delta_combined.flatten()[9:]

    ori_dc, ori_sg, ori_cv = delta_wl(match_wl, target_wl)
    new_dc, new_sg, new_cv = delta_wl(subset_delta_wl, target_wl)

    ori_del15, ori_del45, ori_del110, ori_del_total = delta_e_3angle(match_lab, target_lab, flag_code)
    new_del15, new_del45, new_del110, new_del_total = delta_e_3angle(subset_delta_lab, target_lab, flag_code)

    _, _, target_15H = colour.Lab_to_LCHab(target_lab[:3].values)
    _, _, target_45H = colour.Lab_to_LCHab(target_lab[3:6].values)
    _, _, target_110H = colour.Lab_to_LCHab(target_lab[6:].values)

    _, _, new_15H = colour.Lab_to_LCHab(subset_delta_lab[:3])
    _, _, new_45H = colour.Lab_to_LCHab(subset_delta_lab[3:6])
    _, _, new_110H = colour.Lab_to_LCHab(subset_delta_lab[6:])

    # new_score = (new_del15 - ori_del15) * weights[0] + (new_del45 - ori_del45) * weights[1] + \
    #             (new_del110 - ori_del110) * weights[2] + (new_dc - ori_dc) * weights[3] + \
    #             (new_sg - ori_sg) * weights[4] + (new_cv - ori_cv) * weights[5]
    new_score = new_del15 * weights[0] + new_del45 * weights[1] + \
                new_del110 * weights[2] + new_dc * weights[3] + \
                new_sg * weights[4] + new_cv * weights[5]

    score = Utils.metal_level_score.calculate_score_with_weights(new_del_total, new_dc, new_sg, new_cv,
                                                                 abs(target_15H - new_15H),
                                                                 abs(target_45H - new_45H),
                                                                 abs(target_110H - new_110H))
    if np.isnan(new_del_total) or np.isnan(new_dc) or np.isnan(new_sg) or np.isnan(new_cv):
        delta_sum = float('inf')
    else:
        delta_sum = new_del_total + new_dc + new_sg + new_cv
    if (new_del15 < ori_del15) and (new_del45 < ori_del45) and (new_del110 < ori_del110) and (
            new_del_total < ori_del_total):
        return True, new_del_total, new_dc, new_sg, new_cv, score, new_score, delta_sum
    return False, new_del_total, new_dc, new_sg, new_cv, score, new_score, delta_sum


def analysis_subset_deltaE_trend(model, df_delta_colorants, target_lab, match_lab, weights, flag_code):
    """
    针对素色配方，分析改变逐个色母时的色差变化情况
    :param model: 用来分析当前配方色差的模型
    :param df_delta_colorants: 当前配方
    :param target_lab: 目标的颜色信息
    :param match_lab: 基础配方的颜色信息
    :param weights: 权重
    :return: 3角度ΔE是否都减小，新的色差，两种分数
    """

    subset_delta_lab = model.predict(df_delta_colorants.values)
    subset_delta_lab = subset_delta_lab.flatten()

    ori_del15, ori_del45, ori_del110, ori_del_total = delta_e_3angle(match_lab, target_lab, flag_code)
    new_del15, new_del45, new_del110, new_del_total = delta_e_3angle(subset_delta_lab, target_lab, flag_code)

    _, _, target_15H = colour.Lab_to_LCHab(target_lab[:3].values)
    _, _, target_45H = colour.Lab_to_LCHab(target_lab[3:6].values)
    _, _, target_110H = colour.Lab_to_LCHab(target_lab[6:].values)

    _, _, new_15H = colour.Lab_to_LCHab(subset_delta_lab[:3])
    _, _, new_45H = colour.Lab_to_LCHab(subset_delta_lab[3:6])
    _, _, new_110H = colour.Lab_to_LCHab(subset_delta_lab[6:])

    # new_score = (new_del15 - ori_del15) * weights[0] + (new_del45 - ori_del45) * weights[1] + \
    #             (new_del110 - ori_del110) * weights[2]
    new_score = new_del15 * weights[0] + new_del45 * weights[1] + new_del110 * weights[2]

    score = Utils.solid_level_score.calculate_score_with_weights(new_del_total, '', None, '', None,
                                                                 abs(target_15H - new_15H),
                                                                 abs(target_45H - new_45H),
                                                                 abs(target_110H - new_110H))
    if (new_del15 < ori_del15) and (new_del45 < ori_del45) and (new_del110 < ori_del110) and (
            new_del_total < ori_del_total):
        return True, new_del_total, score, new_score
    return False, new_del_total, score, new_score


def optimal_score(scores, origin_score, score_diff_threshold=0.2):
    """
    根据分数，寻找微调色母输出的最佳个数
    """
    combined = [(score[0], score[1], index) for index, score in enumerate(scores)]
    combined.sort(key=lambda x: (-x[0], x[2]))
    last_score = max(score[0] for score in scores)
    min_score_index = float('inf')
    final_score = None
    final_diff_score = None

    for score, diff_score, index in combined:
        if abs(last_score - score) <= score_diff_threshold:
            if index < min_score_index:
                min_score_index = index
                final_score = score
                final_diff_score = diff_score
            elif index == min_score_index and score > final_score:
                final_score = score
                final_diff_score = diff_score

    if final_score > origin_score and final_score != last_score:
        return final_score, final_diff_score, min_score_index + 1
    else:
        max_scores = [score for score in scores if score[0] == last_score]
        min_score = min(max_scores, key=lambda x: x[1])
        return min_score[0], min_score[1], scores.index(min_score) + 1


def compute_weight_with_target_diff(df_delta, target, match, x_pure_weights, x_particle_weights=None):
    """
    计算 lab_15,lab_45,lab_110,dc,sg,cv 的权重，仅适用于最小二乘线性回归，已知x_weights
    :param df_delta: 匹配数据集
    :param target: 目标
    :param match: 基础配方
    :param x_pure_weights: 考虑颜色属性时df_delta的权重
    :param x_particle_weights: 考虑纹理属性时df_delta的权重，素色配方不考虑纹理属性，因此默认为None
    :return: lab_15,lab_45,lab_110,dc,sg,cv对应的权重列表
    """

    lab = ["Angle15L", "Angle15A", "Angle15B",
           "Angle45L", "Angle45A", "Angle45B",
           "Angle110L", "Angle110A", "Angle110B"]
    wl_feature = ['dc', 'sg15', 'sgm15', 'sg45', 'sg80', 'cv15', 'cvm15', 'cv45', 'cv80']

    # 计算差异
    delta_diff_values = [delta_e_3angle_list(df_delta.loc[i, lab], target[lab], match['flag_code'])
                         for i in range(len(df_delta))]
    target_match_diff = delta_e_3angle_list(match[lab], target[lab], match['flag_code'])
    weight_list = [x_pure_weights for _ in range(3)]

    if x_particle_weights is None:
        columns = ['delta_lab_15', 'delta_lab_45', 'delta_lab_110']
    else:
        columns = ['delta_lab_15', 'delta_lab_45', 'delta_lab_110', 'delta_dc', 'delta_sg', 'delta_cv']
        delta_diff_values = [
            delta_e_3angle_list(df_delta.loc[i, lab], target[lab], match['flag_code']) +
            delta_wl(df_delta.loc[i, wl_feature], target[wl_feature])
            for i in range(len(df_delta))]
        target_match_diff = target_match_diff + delta_wl(match[wl_feature], target[wl_feature])
        weight_list = weight_list + [x_particle_weights for _ in range(3)]

    delta_diff_values = list(map(list, zip(*delta_diff_values)))

    # 计算权重
    weights = []
    for i in range(len(delta_diff_values)):
        numerator = sum(abs(a) * abs(b) for a, b in zip(delta_diff_values[i], weight_list[i]))
        denominator = sum(abs(x) for x in weight_list[i])
        avg_diff = float(numerator) / float(denominator)
        if avg_diff == 0:
            weight = 0
        else:
            weight = target_match_diff[i] / avg_diff
        if math.isnan(weight):
            weight = 1
        weights.append(weight)
    # weights_dict = dict(zip(columns, weights))
    # print("weights:")
    # print(weights_dict)
    return weights


def white_pearls_process(df, formula):
    """
    针对白珍珠配方的特殊处理：将只包含白珍珠的配方的cv设为0
    """
    WHITE_P = ['98-M919', '11-E-014', '93-M011', '93-M010', '11-E-025', '11-E-435']

    count = 0
    for index, row in df.iterrows():
        if all(row[col] == 0 for col in formula if col not in WHITE_P):
            count += 1
            df.at[index, 'cv'] = 0
    return df


def merge_colorants(result, three=False):
    """
    根据人工经验对微调色母最终的输出进行筛选
    """
    colorant_dict = {
        'M': ['90-905', '90-M99/00', '90-M99/01', '90-M99/02', '90-M99/03', '90-M99/04', '90-M99/23', '90-M99/24'],
        'P': ['11-E-014', '11-E-025', '11-E-120', '11-E-220', '11-E-280', '11-E-330',
              '11-E-435', '11-E-440', '11-E-460', '11-E-480', '11-E-520', '11-E-620',
              '11-E-630', '11-E-650', '11-E-660', '11-E-680', '11-E-830', '11-E-850',
              '11-E-910', '11-E-920', '93-M010', '93-M011', '93-M176', '93-M363',
              '93-M364', '93-M505', '93-M506', '98-M319', '98-M919', '98-M930'
              ],
        'black': ['90-A927', '90-A924', '90-A926', '90-A997', '90-1250'],
        'white': ['98-A097', '90-A032', '90-A035', '90-A031'],
        'green': ['90-A695', '90-A640'],
        'red': ['90-A350', '90-A378', '90-A306', '90-3A0', '90-A359', '90-A323', '90-A307', '90-A372', '90-A347'],
        'blue': ['90-A528', '90-A589', '90-A563', '90-A503', '90-A527'],
        'yellow': ['90-A143', '90-A177', '90-A148', '90-A115', '90-A105', '90-A136', '90-A155', '90-A149'],
        'orange': ['90-A201', '90-A329'],
        'purple': ['90-A427', '90-A430']
    }

    if three:
        for key in colorant_dict:
            colorant_dict[key] = [item + suffix for item in colorant_dict[key] for suffix in ['_c', '_p']]

    merged_result = []
    added_colorants = set()
    found_M_colorant = False
    black_colorant = None
    white_colorant = None
    for colorant, delta in result:
        for color_type, colorant_list in colorant_dict.items():
            if colorant in colorant_list and colorant not in added_colorants:
                if color_type == 'white':
                    white_colorant = (colorant, delta)
                if color_type == 'black':
                    black_colorant = (colorant, delta)
                merged_result.append((colorant, delta))
                added_colorants.update(colorant_list)
                # if color_type == 'M' or color_type == 'P':
                #     merged_result.append((colorant, delta))
                #     added_colorants.add(colorant)
                #     if color_type == 'M':
                #         found_M_colorant = True
                # elif not found_M_colorant or (found_M_colorant and color_type != 'white'):
                #     if color_type == 'white':
                #         white_colorant = (colorant, delta)
                #     if color_type == 'black':
                #         black_colorant = (colorant, delta)
                #     merged_result.append((colorant, delta))
                #     added_colorants.update(colorant_list)
                # break
    if black_colorant is not None and white_colorant is not None:
        if (black_colorant[1] * white_colorant[1] < 0 and black_colorant[1] > 0) or \
                (black_colorant[1] * white_colorant[1] > 0 and result.index(white_colorant) < result.index(
                    black_colorant)):
            merged_result.remove((black_colorant[0], black_colorant[1]))
        else:
            merged_result.remove((white_colorant[0], white_colorant[1]))

    return merged_result


def is_subset(row, target_set, threshold=0.8):
    row_set = set(row[row != 0].index.tolist())
    common_elements = row_set.intersection(target_set)
    return len(common_elements) >= math.floor(len(target_set) * threshold)


def is_intersect(row, target_set):
    row_set = set(row[row != 0].index.tolist())
    common_elements = row_set.intersection(target_set)
    return len(common_elements) >= 1


def predict_colorants_with_particlde(df_delta_formula, df_delta_wl, df_delta_lab, target_delta_wl, target_delta_lab,
                                     particle_colorants, pure_colorants, formula):
    # 首先，计算 delta_particle_colorants
    # df_delta_wl = df_delta[wl]      df_delta[formula] = df_delta_formula   df_delta_lab = df_delta[lab]
    x_particle, _, _, _ = np.linalg.lstsq(df_delta_wl.values.T, target_delta_wl.values.T, rcond=None)
    delta_colorants_particle = np.dot(df_delta_formula.values.T, x_particle)
    indices_particle = [formula.index(colorant) for colorant in particle_colorants]
    delta_particle_colorants = delta_colorants_particle[indices_particle]

    # 定义一个损失函数，该函数将最小化 df_delta[lab].values.T*x_pure - target_delta_lab 的平方和，
    # 同时确保 delta_colorants 中的 particle_colorants 部分与 delta_particle_colorants 一致
    def loss(x_pure):
        delta_colorants = np.dot(df_delta_formula.values.T, x_pure)
        delta_colorants[indices_particle] = delta_particle_colorants
        return np.sum((np.dot(df_delta_lab.values.T, x_pure) - target_delta_lab) ** 2) + np.sum(
            (delta_colorants[indices_particle] - delta_particle_colorants) ** 2)

    # 使用 scipy 的 minimize 函数来找到最小化损失函数的 x_pure
    result = minimize(loss, x_particle, method='BFGS')

    # 最后计算 delta_colorants
    x_pure = result.x
    delta_colorants = np.dot(df_delta_formula.values.T, x_pure)
    return delta_colorants, x_pure


def run_strategy(df_delta, match, target, colorants, fit_target, reflectance_coef=None):
    df_delta[colorants] = (df_delta[colorants] - match[colorants]).astype(float)
    target_fit_target = target[fit_target] - match[fit_target]
    df_delta[fit_target] = (df_delta[fit_target] - match[fit_target]).astype(float)

    df_delta[fit_target] = df_delta[fit_target].apply(pd.to_numeric, errors='coerce')
    target_fit_target = target_fit_target.apply(pd.to_numeric, errors='coerce')
    if reflectance_coef is None:
        extended_matrix = df_delta[fit_target].values
        extended_target = target_fit_target.values
    else:
        def string_list_to_float_list(s):
            if isinstance(s, str):
                s = s.strip('[]')
                elements = s.split(',')
                return [float(x.strip()) for x in elements]
            else:
                return s

        for coef in reflectance_coef:
            df_delta[coef] = df_delta[coef].apply(string_list_to_float_list)
            target[coef] = string_list_to_float_list(target[coef])
            match[coef] = string_list_to_float_list(match[coef])

        target_delta_refl_coef = []
        for coef in reflectance_coef:
            delta_list = [t - m for t, m in zip(target[coef], match[coef])]
            target_delta_refl_coef.extend(delta_list)
        target_delta_refl_coef_np = \
            np.array(target_delta_refl_coef).reshape(-1, len(reflectance_coef) * len(target[reflectance_coef[0]]))

        df_delta_refl_coef_matrix = []
        for _, row in df_delta.iterrows():
            row_diffs = [a - b for coef in reflectance_coef for a, b in zip(row[coef], match[coef])]
            df_delta_refl_coef_matrix.append(row_diffs)
        df_delta_refl_coef_matrix = np.array(df_delta_refl_coef_matrix)

        extended_matrix = np.hstack((df_delta_refl_coef_matrix, df_delta[fit_target].values))
        extended_target = np.hstack(
            (target_delta_refl_coef_np, target_fit_target.values.reshape(1, -1)))

    x, _, _, _ = np.linalg.lstsq(extended_matrix.T, extended_target.T, rcond=None)
    delta_colorants = np.dot(df_delta[colorants].values.T, x)
    return delta_colorants, x, df_delta


def predict_particle(target, match, df, match_columns, pure_colorants, lab, reflectance_coef, particle_colorants, wl,
                     formula, ignore_colorants, more_info):
    """
    针对颗粒配方的调色策略
    :param target: 目标
    :param match: 基础配方
    :param df: 匹配数据集
    :param match_columns: 基础配方中所含的色母
    :param pure_colorants: 素色色母列表
    :param lab: 颜色属性
    :param particle_colorants: 颗粒色母列表
    :param wl: 纹理属性
    :param formula: 全部色母列表
    :param ignore_colorants: 忽略的色母列表
    :return: 微调色母及对应的改变量
    """
    # 获取信息
    white_pearls_process(df, formula)
    white_pearls_process(match, formula)
    target = target.iloc[0, :]
    match = match.iloc[0, :]

    _, _, target_15H = colour.Lab_to_LCHab(target[lab[:3]].values)
    _, _, target_45H = colour.Lab_to_LCHab(target[lab[3:6]].values)
    _, _, target_110H = colour.Lab_to_LCHab(target[lab[6:]].values)

    _, _, match_15H = colour.Lab_to_LCHab(match[lab[:3]].values)
    _, _, match_45H = colour.Lab_to_LCHab(match[lab[3:6]].values)
    _, _, match_110H = colour.Lab_to_LCHab(match[lab[6:]].values)

    deltaE = delta_e(target[lab], match[lab], match['flag_code'])
    deltaWL = delta_wl(target[wl], match[wl])

    score = Utils.metal_level_score.calculate_score_with_weights(deltaE, deltaWL[0], deltaWL[1],
                                                                 deltaWL[2], abs(target_15H - match_15H),
                                                                 abs(target_45H - match_45H),
                                                                 abs(target_110H - match_110H))

    # 对颗粒色母进行限制
    non_zero_columns = match[particle_colorants][match[particle_colorants] != 0].index.tolist()
    df_subset = df[
        df.apply(lambda row: set(row[particle_colorants][row[particle_colorants] != 0].index.tolist()) == set(
            non_zero_columns), axis=1)]

    if len(df_subset) < 20:
        df_subset = df[df.apply(lambda row: set(row[particle_colorants][row[particle_colorants] != 0].index.tolist()).
                                issubset(set(non_zero_columns)), axis=1)]
        if len(df_subset) < 20:
            df_subset = df[df.apply(lambda row: is_subset(row[particle_colorants], set(non_zero_columns)), axis=1)]
            if len(df_subset) < 20:
                df_subset = df[
                    df.apply(lambda row: is_intersect(row[particle_colorants], set(non_zero_columns)), axis=1)]
                if len(df_subset) < 20:
                    df_subset = df.copy()
    #             else:
    #                 print("配方总数量: {}  颗粒色母与基础色母有交集的配方数量: {}".format(len(df), len(df_subset)))
    #         else:
    #             print("配方总数量: {}  颗粒色母包含80%基础色母的配方数量: {}".format(len(df), len(df_subset)))
    #     else:
    #         print("配方总数量: {}  颗粒色母是基础配方颗粒色母子集的配方数量: {}".format(len(df), len(df_subset)))
    # else:
    #     print("配方总数量: {}  颗粒色母完全一致的配方数量: {}".format(len(df), len(df_subset)))
    #
    # print("颗粒配方总数量: {}".format(len(df_subset)))

    df_subset.reset_index(drop=True, inplace=True)
    df_delta = df_subset.copy()

    # 调色策略
    delta_colorants, x_weights, df_delta = run_strategy(df_delta, match, target, formula, lab + wl + more_info,
                                                        reflectance_coef)

    weights = compute_weight_with_target_diff(df_subset, target, match, x_weights, x_weights)
    deltaEs = delta_e_3angle_list(target[lab], match[lab], match['flag_code'])
    new_score = deltaEs[0] * weights[0] + deltaEs[1] * weights[1] + deltaEs[2] * weights[2] + \
                deltaWL[0] * weights[3] + deltaWL[1] * weights[4] + deltaWL[2] * weights[5]
    # print("delta_e_all: {}, delta_e: {}, delta_wl: {}, score: {}, new_score: {}".format(deltaE, deltaEs, deltaWL,
    #                                                                                     score, new_score))

    feature_order = sort_colorants_result(df_delta, formula, lab + wl, match_columns)

    best_subset, best_sign_match = find_best_subset_order_method(df_subset, delta_colorants, formula,
                                                                 feature_order, match_columns, match,
                                                                 wl, target[wl], match[wl],
                                                                 lab, target[lab], match[lab],
                                                                 score, ignore_colorants, weights, True)
    # if best_sign_match:
    #     print(best_subset)
    # else:
    #     print("无法微调")

    # 结果输出
    final_colorants = list(set(best_subset) & set(match_columns))

    sorted_zipped_formula = sorted(zip(formula, delta_colorants, match[formula]),
                                   key=lambda x: feature_order.index(x[0]))
    result_formula = [(colorant, "add" if delta > 0 else "reduce") for (colorant, delta, origin)
                      in sorted_zipped_formula if abs(delta) > 0 and colorant in final_colorants]
    result_formula_specific = [(colorant, round(float(delta), 4)) for (colorant, delta, origin) in
                               sorted_zipped_formula if abs(delta) > 0 and colorant in final_colorants]
    result_formula_specific_all = [(colorant, round(float(delta), 4)) for (colorant, delta, origin) in
                                   sorted_zipped_formula if colorant in match_columns]

    # print(result_formula)
    # print(result_formula_specific_all)
    # print("---------------------------------")
    return result_formula_specific, result_formula_specific_all


def predict_pure(target, match, df, match_columns, pure_colorants, lab, reflectance_coef, ignore_colorants, more_info):
    """
    针对素色配方的调色策略
    :param target: 目标
    :param match: 基础配方
    :param df: 匹配数据集
    :param match_columns: 基础配方中所含的色母
    :param pure_colorants: 素色色母列表
    :param lab: 颜色属性
    :param ignore_colorants: 忽略的色母列表
    :return: 微调色母及对应的改变量
    """
    # 获取信息
    target = target.iloc[0, :]
    match = match.iloc[0, :]

    _, _, target_15H = colour.Lab_to_LCHab(target[lab[:3]].values)
    _, _, target_45H = colour.Lab_to_LCHab(target[lab[3:6]].values)
    _, _, target_110H = colour.Lab_to_LCHab(target[lab[6:]].values)

    _, _, match_15H = colour.Lab_to_LCHab(match[lab[:3]].values)
    _, _, match_45H = colour.Lab_to_LCHab(match[lab[3:6]].values)
    _, _, match_110H = colour.Lab_to_LCHab(match[lab[6:]].values)

    deltaE = delta_e(target[lab], match[lab], match['flag_code'])
    score = Utils.solid_level_score.calculate_score_with_weights(deltaE, '', None, '', None,
                                                                 abs(target_15H - match_15H),
                                                                 abs(target_45H - match_45H),
                                                                 abs(target_110H - match_110H))
    df_subset = df.copy()

    df_subset.reset_index(drop=True, inplace=True)
    df_delta = df_subset.copy()

    # 调色策略
    delta_pure_colorants, x_pure, df_delta = run_strategy(df_delta, match, target, pure_colorants, lab + more_info,
                                                          reflectance_coef)

    # weights = compute_weight_with_diff_matrix(df_subset, target, match)
    weights = compute_weight_with_target_diff(df_subset, target, match, x_pure)
    deltaEs = delta_e_3angle_list(target[lab], match[lab], match['flag_code'])
    new_score = deltaEs[0] * weights[0] + deltaEs[1] * weights[1] + deltaEs[2] * weights[2]
    # print("delta_e_all: {}, delta_e: {}, score: {}, new_score: {}".format(deltaE, deltaEs, score, new_score))

    feature_order = sort_colorants_result(df_delta, pure_colorants, lab, match_columns)

    best_subset, best_sign_match = find_best_subset_order_method(df_subset, delta_pure_colorants,
                                                                 pure_colorants,
                                                                 feature_order, match_columns, match,
                                                                 None, None, None,
                                                                 lab, target[lab], match[lab],
                                                                 score, ignore_colorants, weights, False)
    # if best_sign_match:
    #     print(best_subset)
    # else:
    #     print("无法微调")

    # 结果输出
    final_colorants = list(set(best_subset) & set(match_columns))

    sorted_zipped_pure = sorted(zip(pure_colorants, delta_pure_colorants, match[pure_colorants]),
                                key=lambda x: feature_order.index(x[0]))
    result_pure = [(colorant, "add" if delta > 0 else "reduce") for (colorant, delta, origin)
                   in sorted_zipped_pure if abs(delta) > 0 and colorant in final_colorants]
    result_pure_specific = [(colorant, round(float(delta), 4)) for (colorant, delta, origin) in sorted_zipped_pure
                            if abs(delta) > 0 and colorant in final_colorants]
    result_pure_specific_all = [(colorant, round(float(delta), 4)) for (colorant, delta, origin) in
                                sorted_zipped_pure if colorant in match_columns]

    # print(result_pure)
    # print(result_pure_specific_all)
    # print("---------------------------------")
    return result_pure_specific, result_pure_specific_all


# more_dataset = pd.read_excel("../data/dws_formula_gauge_with_texture.xlsx")
# more_dataset['particle_combine_all'] = more_dataset['particle_combine_all'].str.split(',')
# more_dataset_dict_raw = more_dataset.set_index('FORMULA_CODE')['particle_combine_all']
# more_dataset_dict_three = {
#     code: particles
#     for code, particles in more_dataset_dict_raw.to_dict().items()
#     if any(item.endswith('_P') for item in particles)
# }
# more_dataset_dict_two = {
#     code: particles
#     for code, particles in more_dataset_dict_raw.to_dict().items()
#     if not any(item.endswith('_P') for item in particles)
# }
# match_dataset = pd.read_excel("../data/match_dataset1.xlsx")
# similar_dataset_dict = np.load('../data/similar_dataset_dict_分工序.npy', allow_pickle=True).item()


def get_match_result(serialNo, match_dataset):
    if not match_dataset[match_dataset['target_code'] == serialNo].empty:
        match_result_str = match_dataset[match_dataset['target_code'] == serialNo]['match_result'].iloc[0]
        try:
            match_df_code = ast.literal_eval(match_result_str)
            return match_df_code
        except (ValueError, SyntaxError) as e:
            print(f"转换 match_result 失败: {e}")
    return []


def expand_data(serialNo, match_request, target_tuple_list, match_tuple_list, target_match_request, match_formula,
                match_particles, particle_colorants, layer_num, more_dataset_dict_three, more_dataset_dict_two,
                match_dataset, similar_dataset_dict, min_num_threshold=30, max_num_threshold=50):
    """
    数据扩充优先级：
    [0]. target_df + match_df, match_diff_0, (target_df + match_df)_diff_0，得到此时的色母并集union1
    [1]. (target_df + match_df + match)_diff_1_2_3,且是并集union1的子集
    [2]. target_df_2 + match_df_2, (target_df_2 + match_df_2)_diff_0, 得到此时的色母并集union2
    [3]. (target_df_2 + match_df_2)_diff_1_2_3,且是并集union2的子集
    [4]. (target_df + match_df + match)_diff_1_2_3   +   (target_df_2 + match_df_2)_diff_1_2_3，按照与色母交集长度排序
    """
    # TODO: 从表中获取基础配方的匹配数据集，没有找到的话调用匹配接口
    match_df_code = []
    if len(match_tuple_list) == 0:
        if serialNo:
            if serialNo in match_dataset['target_code'].values:
                match_df_code = get_match_result(serialNo, match_dataset)
            else:
                match_df_code, _ = Get_Match_Code(serialNo)
        elif match_request:
            match_df_code, _ = Get_Match_Code_with_request_param_id(match_request)
    else:
        match_df_code = [tup[0] for tup in match_tuple_list]

    target_df_code = [tup[0] for tup in target_tuple_list]
    if len(target_tuple_list) == 0:
        target_df_code, _ = Get_Match_Code_with_request_param_id(target_match_request)

    match_df_code = match_df_code[:100]
    target_df_code = target_df_code[:100]

    particle_dict = more_dataset_dict_three if layer_num == 3 else more_dataset_dict_two
    existing_codes = set(similar_dataset_dict.keys()) & set(particle_dict.keys())

    combined_codes = [code for code in set(match_df_code + target_df_code) if code in existing_codes]

    match_formula_code = set(combined_codes)
    match_particles_col = match_particles.intersection(particle_colorants)
    more_formula_code_merge = set()

    for code in combined_codes:
        more_codes = similar_dataset_dict[code]
        match_formula_code.update(more_codes[0])
        for sublist in more_codes[1:]:
            more_formula_code_merge.update(sublist)

    match_formula_code = set(existing_codes) & set(match_formula_code)

    if len(match_formula_code) == 0:
        combined_codes = set([code for code in particle_dict
                              if set(particle_dict[code]).issubset(match_particles)])

    if len(match_formula_code) < min_num_threshold:
        # 二次匹配
        result_code_1 = set(code for code in combined_codes if code in match_dataset['target_code'].values)
        result_code_2 = set()
        for code in result_code_1:
            result_code_2.update(get_match_result(code, match_dataset))
        more_formula_code_merge.update(result_code_2)
        for code in result_code_2:
            if code in existing_codes:
                more_codes = similar_dataset_dict[code]
                for sublist in more_codes:
                    more_formula_code_merge.update(sublist)

        more_formula_code_merge = set(existing_codes) & set(more_formula_code_merge)
        all_union_particles_1 = set().union(*[particle_dict[code] for code in combined_codes])
        match_formula_code.update(set([code for code in more_formula_code_merge
                                       if set(particle_dict[code]).issubset(all_union_particles_1)]))
        if len(match_formula_code) < min_num_threshold:
            match_formula_code.update(more_formula_code_merge)

            if len(match_formula_code) < min_num_threshold:
                # 当前涉及到的全部色母并集
                all_union_particles_2 = set().union(*[particle_dict[code] for code in match_formula_code])
                if len(match_formula_code) < min_num_threshold:
                    code_counts = []
                    for code in (set(particle_dict.keys()) - match_formula_code):
                        temp_particles = set(particle_dict[code])
                        if temp_particles.issubset(all_union_particles_2):
                            match_formula_code.add(code)
                        else:
                            count1 = len(temp_particles.intersection(match_particles)) + \
                                     len(temp_particles.intersection(match_particles_col))
                            count2 = len(temp_particles.intersection(all_union_particles_1))
                            count3 = len(temp_particles.intersection(all_union_particles_2))
                            code_counts.append((code, count1, count2, count3))
                    code_counts.sort(key=lambda x: (x[1], x[2], x[3], x[0]), reverse=True)

                    for code, _, _, _ in code_counts:
                        match_formula_code.add(code)

                    match_formula_code_list = list(match_formula_code)
                    match_formula_code = set(match_formula_code_list[:max_num_threshold])

    if len(match_formula_code) > max_num_threshold * 1.5:
        code_counts = []
        for code in match_formula_code:
            temp_particles = set(particle_dict[code])
            count = len(temp_particles.intersection(match_particles)) + \
                    len(temp_particles.intersection(match_particles_col))
            code_counts.append((code, count))
        code_counts.sort(key=lambda x: (x[1], x[0]), reverse=True)
        selected_codes = [code for code, _ in code_counts[:int(max_num_threshold * 1.5)]]

        match_formula_code = selected_codes
        match_formula_code.extend(combined_codes)
        match_formula_code = set(match_formula_code)

    if serialNo in match_formula_code:
        match_formula_code.remove(serialNo)

    return list(match_formula_code)


def get_match_formula_detail(serialNo, id):
    """
    获取基础配方的详细信息，直接从数据库中获取，保证数据最新
    """

    match, flag, particle_list = sql_data.Get_match_SQL_Data(serialNo, id)
    layer_num = 3 if flag else 2
    # print(f"{layer_num}工序")
    if len(match) == 0:
        return None, None, None
    if 'flag_code' not in match.columns or match['flag_code'] is None:
        match['flag_code'] = get_flag_code(particle_list)
    return match, layer_num, particle_list


def extract_angle_data(sub_request, angle_index, keys_to_extract):
    """
    从sub_request中提取指定角度的数据。
    """
    angle_data = sub_request['colorPanel']['angles'][angle_index]
    return {key: angle_data[key] for key in keys_to_extract if key in angle_data}


def get_coef(ref):
    if isinstance(ref, str):
        Y = ast.literal_eval(ref)
    else:
        Y = ref
    X = np.linspace(400, 700, 31)
    coef = np.polyfit(X, Y, 5)
    return coef.tolist()


def analysis_sub_request(sub_request, type, formula, wl, lab):
    reflectance_15 = sub_request['colorPanel']['angles'][1]['reflectance']
    reflectance_45 = sub_request['colorPanel']['angles'][2]['reflectance']
    reflectance_110 = sub_request['colorPanel']['angles'][3]['reflectance']

    reflectance_15_coef = get_coef(reflectance_15)
    reflectance_45_coef = get_coef(reflectance_45)
    reflectance_110_coef = get_coef(reflectance_110)

    lab_15 = Refl_2_Lab(reflectance_15)
    lab_45 = Refl_2_Lab(reflectance_45)
    lab_110 = Refl_2_Lab(reflectance_110)

    keys_to_extract = ['sg', 'cv', 'sa', 'si', 'sd', 'simple_coarseness', 'background_coarseness',
                       'directional_coarseness']

    angle_data_15 = extract_angle_data(sub_request, 1, keys_to_extract)
    angle_data_m15 = extract_angle_data(sub_request, 0, keys_to_extract)
    angle_data_45 = extract_angle_data(sub_request, 2, keys_to_extract)
    angle_data_80 = extract_angle_data(sub_request, 3, keys_to_extract)

    reflectance_coef = ['reflectance15_6coef', 'reflectance45_6coef', 'reflectance110_6coef']
    wl = ['dc', 'sg15', 'sgm15', 'sg45', 'sg80', 'cv15', 'cvm15', 'cv45', 'cv80',
          'sam15', 'sa15', 'sa45', 'sa80', 'sim15', 'si15', 'si45', 'si80', 'simple_coarsenessM15',
          'simple_coarseness15', 'simple_coarseness45', 'simple_coarseness80',
          'background_coarseness15', 'directional_coarseness15', 'sd15',
          'background_coarsenessM15', 'directional_coarsenessM15', 'sdm15',
          'background_coarseness45', 'directional_coarseness45', 'sd45',
          'background_coarseness80', 'directional_coarseness80', 'sd80'
          ]

    detail = pd.DataFrame(columns=lab + wl + reflectance_coef)
    detail_value = {
        'dc': sub_request['dc'],
        'sg15': angle_data_15['sg'],
        'sgm15': angle_data_m15['sg'],
        'sg45': angle_data_45['sg'],
        'sg80': angle_data_80['sg'],
        'cv15': angle_data_15['cv'],
        'cvm15': angle_data_m15['cv'],
        'cv45': angle_data_45['cv'],
        'cv80': angle_data_80['cv'],
        'sa15': angle_data_15['sa'],
        'sam15': angle_data_m15['sa'],
        'sa45': angle_data_45['sa'],
        'sa80': angle_data_80['sa'],
        'si15': angle_data_15['si'],
        'sim15': angle_data_m15['si'],
        'si45': angle_data_45['si'],
        'si80': angle_data_80['si'],
        'Angle15L': lab_15[0], 'Angle15A': lab_15[1], 'Angle15B': lab_15[2],
        'Angle45L': lab_45[0], 'Angle45A': lab_45[1], 'Angle45B': lab_45[2],
        'Angle110L': lab_110[0], 'Angle110A': lab_110[1], 'Angle110B': lab_110[2],
        'simple_coarsenessM15': angle_data_m15['simple_coarseness'],
        'simple_coarseness15': angle_data_15['simple_coarseness'],
        'simple_coarseness45': angle_data_45['simple_coarseness'],
        'simple_coarseness80': angle_data_80['simple_coarseness'],
        'background_coarsenessM15': angle_data_m15['background_coarseness'],
        'background_coarseness15': angle_data_15['background_coarseness'],
        'background_coarseness45': angle_data_45['background_coarseness'],
        'background_coarseness80': angle_data_80['background_coarseness'],
        'directional_coarsenessM15': angle_data_m15['directional_coarseness'],
        'directional_coarseness15': angle_data_15['directional_coarseness'],
        'directional_coarseness45': angle_data_45['directional_coarseness'],
        'directional_coarseness80': angle_data_80['directional_coarseness'],
        'sdm15': angle_data_m15['sd'], 'sd15': angle_data_15['sd'],
        'sd45': angle_data_45['sd'], 'sd80': angle_data_80['sd'],
        'reflectance15_6coef': reflectance_15_coef,
        'reflectance45_6coef': reflectance_45_coef,
        'reflectance110_6coef': reflectance_110_coef,
    }

    detail.loc[0] = detail_value
    request_param = {
        "data": [
            sub_request
        ]
    }

    if 'match_result' in sub_request and sub_request['match_result'] is not None:
        dict_list = json.loads(sub_request['match_result'])
        tuple_list = [(item['formulaCode'], item['gaugeId']) for item in dict_list]
    else:
        tuple_list = []

    # 实车
    if type == 0:
        return detail, tuple_list, request_param
    # 打板
    else:
        if 'flag_code' in sub_request and sub_request['flag_code'] is not None:
            detail['flag_code'] = sub_request['flag_code']
        # 返回match, match中的色母, 匹配接口输入
        layer_three = any(item['colorLayer'] == 'P' for item in sub_request['formula_detail'])
        if layer_three:
            formula_suffixes = ['_C', '_P']
            formula = [item + suffix for item in formula for suffix in formula_suffixes]
        for col in formula:
            detail[col] = 0
        match_colorants_col = []

        factor = 200 if layer_three else 100
        total_weight = sum(item['weight'] for item in sub_request['formula_detail'])
        for item in sub_request['formula_detail']:
            new_weight = (item['weight'] / total_weight) * factor

            if layer_three:
                if item['colorLayer'] == 'C':
                    detail.at[0, item['serialNo'] + '_C'] = [new_weight]
                    match_colorants_col.append(item['serialNo'] + '_C')
                else:
                    detail.at[0, item['serialNo'] + '_P'] = [new_weight]
                    match_colorants_col.append(item['serialNo'] + '_P')
            else:
                detail.at[0, item['serialNo']] = [new_weight]
                match_colorants_col.append(item['serialNo'] + '_C')

        layer_num = 3 if layer_three else 2
        if 'flag_code' not in detail.columns or detail['flag_code'] is None:
            detail['flag_code'] = get_flag_code(match_colorants_col)
        return detail, match_colorants_col, request_param, layer_num, tuple_list


def get_flag_code(particle_list):
    # 素色S-36种  珍珠P-30种  银粉M-8种   排除'90-M1', '90-M4', '90-M5'
    S_set = {'90-3A0', '90-A031', '90-A032', '90-A035', '90-A105', '90-A115', '90-A136', '90-A143', '90-A148',
             '90-A149', '90-A155', '90-A177', '90-A201', '90-A306', '90-A307', '90-A323', '90-A329', '90-A347',
             '90-A350', '90-A359', '90-A372', '90-A378', '90-A427', '90-A430', '90-A503', '90-A527', '90-A528',
             '90-A563', '90-A589', '90-A640', '90-A695', '90-A924', '90-A926', '90-A927', '90-A997', '98-A097'}
    P_set = {'11-E-014', '11-E-025', '11-E-120', '11-E-220', '11-E-280', '11-E-330', '11-E-435', '11-E-440', '11-E-460',
             '11-E-480', '11-E-520', '11-E-620', '11-E-630', '11-E-650', '11-E-660', '11-E-680', '11-E-830', '11-E-850',
             '11-E-910', '11-E-920', '93-M010', '93-M011', '93-M176', '93-M363', '93-M364', '93-M505', '93-M506',
             '98-M319', '98-M919', '98-M930'}
    M_set = {'90-905', '90-M99/00', '90-M99/01', '90-M99/02', '90-M99/03', '90-M99/04', '90-M99/23', '90-M99/24'}

    colorants = {item[:-2] if item.endswith('_C') or item.endswith('_P') else item for item in particle_list}

    if colorants.intersection(P_set) and colorants.intersection(M_set):
        return 'MP'
    elif colorants.intersection(P_set) and not colorants.intersection(M_set):
        return 'P'
    elif colorants.intersection(M_set) and not colorants.intersection(P_set):
        return 'M'
    else:
        return 'S'


def run(request, more_dataset_dict_three, more_dataset_dict_two, match_dataset, similar_dataset_dict,
        use_return_list=True):
    start_time = time.time()
    # result = {"C": {}, "P": {}}
    result = {'order': {}, 'num': 0}
    # 素色S-37种  珍珠P-30种  银粉M-8种   排除'90-M1', '90-M4', '90-M5'
    formula = ['11-E-014', '11-E-025', '11-E-120',
               '11-E-220', '11-E-280', '11-E-330',
               '11-E-435', '11-E-440', '11-E-460',
               '11-E-480', '11-E-520', '11-E-620',
               '11-E-630', '11-E-650', '11-E-660',
               '11-E-680', '11-E-830', '11-E-850',
               '11-E-910', '11-E-920', '90-1250',
               '90-3A0', '90-905', '90-A031',
               '90-A032', '90-A035', '90-A105',
               '90-A115', '90-A136', '90-A143',
               '90-A148', '90-A149', '90-A155',
               '90-A177', '90-A201', '90-A306',
               '90-A307', '90-A323', '90-A329',
               '90-A347', '90-A350', '90-A359',
               '90-A372', '90-A378', '90-A427',
               '90-A430', '90-A503', '90-A527',
               '90-A528', '90-A563', '90-A589',
               '90-A640', '90-A695', '90-A924',
               '90-A926', '90-A927', '90-A997',
               '90-M1', '90-M4', '90-M5',
               '90-M99/00', '90-M99/01', '90-M99/02',
               '90-M99/03', '90-M99/04', '90-M99/23',
               '90-M99/24', '93-M010', '93-M011',
               '93-M176', '93-M363', '93-M364',
               '93-M505', '93-M506', '98-A097',
               '98-M319', '98-M919', '98-M930']
    particle_colorants = ['11-E-014', '11-E-025', '11-E-120',
                          '11-E-220', '11-E-280', '11-E-330',
                          '11-E-435', '11-E-440', '11-E-460',
                          '11-E-480', '11-E-520', '11-E-620',
                          '11-E-630', '11-E-650', '11-E-660',
                          '11-E-680', '11-E-830', '11-E-850',
                          '11-E-910', '11-E-920', '93-M010',
                          '93-M011', '93-M176', '93-M363',
                          '93-M364', '93-M505', '93-M506',
                          '98-M319', '98-M919', '98-M930',
                          '90-905',
                          '90-M99/00',
                          '90-M99/01',
                          '90-M99/02',
                          '90-M99/03',
                          '90-M99/04',
                          '90-M99/23',
                          '90-M99/24',
                          ]
    wl = ['dc', 'sg15', 'sgm15', 'sg45', 'sg80', 'cv15', 'cvm15', 'cv45', 'cv80',
          'sam15', 'sa15', 'sa45', 'sa80', 'sim15', 'si15', 'si45', 'si80', 'simple_coarsenessM15',
          'simple_coarseness15', 'simple_coarseness45', 'simple_coarseness80',
          'background_coarseness15', 'directional_coarseness15', 'sd15',
          'background_coarsenessM15', 'directional_coarsenessM15', 'sdm15',
          'background_coarseness45', 'directional_coarseness45', 'sd45',
          'background_coarseness80', 'directional_coarseness80', 'sd80'
          ]
    pure_colorants = ['90-3A0', '90-1250', '90-A031',
                      '90-A032', '90-A035', '90-A105',
                      '90-A115', '90-A136', '90-A143',
                      '90-A148', '90-A149', '90-A155',
                      '90-A177', '90-A201', '90-A306',
                      '90-A307', '90-A323', '90-A329',
                      '90-A347', '90-A350', '90-A359',
                      '90-A372', '90-A378', '90-A427',
                      '90-A430', '90-A503', '90-A527',
                      '90-A528', '90-A563', '90-A589',
                      '90-A640', '90-A695', '90-A924',
                      '90-A926', '90-A927', '90-A997',
                      '98-A097',
                      '90-M1', '90-M4', '90-M5']
    lab = ["Angle15L", "Angle15A", "Angle15B",
           "Angle45L", "Angle45A", "Angle45B",
           "Angle110L", "Angle110A", "Angle110B"]
    ignore_colorants = ['90-M1', '90-M4', '90-M5']
    more_info = []
    reflectance_coef = ['reflectance15_6coef', 'reflectance45_6coef', 'reflectance110_6coef']

    if len(request['data']) == 2:
        serialNo, target, target_tuple_list, match_tuple_list, target_request, match, match_colorants_col, match_request, layer_num = \
            None, None, [], [], None, None, None, None, None
        for item in request['data']:
            if 'formula_detail' not in item or item['formula_detail'] is None:
                target, target_tuple_list, target_request = analysis_sub_request(item, 0, formula, wl, lab)
                if not use_return_list:
                    target_tuple_list = []
            else:
                match, match_colorants_col, match_request, layer_num, match_tuple_list = \
                    analysis_sub_request(item, 1, formula, wl, lab)
                serialNo = None
    else:
        match_request = None
        target, target_tuple_list, target_request = analysis_sub_request(request['data'][0], 0, formula, wl, lab)
        serialNo = request['data'][0]['serialNo']
        if 'gaugeId' in request['data'][0]:
            gaugeId = request['data'][0]['gaugeId']
        else:
            gaugeId = None
        match, layer_num, match_colorants_col = get_match_formula_detail(serialNo, gaugeId)
        match_tuple_list = []

    if match is None or target is None or (serialNo is None and match_request is None and len(match_tuple_list) == 0) \
            or (len(target_tuple_list) == 0 and target_request is None):
        # print("没有找到配方")
        return result
    formula_suffixes = ['_C', '_P'] if layer_num == 3 else ['']
    formula = [item + suffix for item in formula for suffix in formula_suffixes]
    particle_colorants = [item + suffix for item in particle_colorants for suffix in formula_suffixes]
    pure_colorants = [item + suffix for item in pure_colorants for suffix in formula_suffixes]
    ignore_colorants = [item + suffix for item in ignore_colorants for suffix in formula_suffixes]

    mask = (match[formula] != 0).any()
    match_columns = match[formula].columns[mask].tolist()
    # print("基础配方包含的色母:", match_columns)
    start_time1 = time.time()

    match_formula_code = expand_data(serialNo, match_request, target_tuple_list, match_tuple_list,
                                     target_request, match, set(match_colorants_col),
                                     particle_colorants, layer_num, more_dataset_dict_three,
                                     more_dataset_dict_two, match_dataset, similar_dataset_dict)

    # start_time2 = time.time()
    # print("数据扩充：", start_time2 - start_time1)
    if len(match_formula_code) + len(match_tuple_list) == 0:
        # print("匹配配方数据集长度为0")
        return result

    # TODO:从本地表中获取最终的匹配数据集的详细信息
    if layer_num == 3:
        # print("三工序")
        match_df = sql_data.Get_match_dataset_SQL_Data_With_Code(match_formula_code, True)
    else:
        # print("二工序")
        match_df = sql_data.Get_match_dataset_SQL_Data_With_Code(match_formula_code, False)
    # start_time3 = time.time()
    # print("从数据库中获取数据：", start_time3 - start_time2)
    if match_df.empty:
        # print("没有找到匹配配方")
        return result
    # print("最终得到的匹配配方数量:", len(match_df))
    particle_num = get_formula_type(match, particle_colorants)

    if particle_num == 0:
        # print("素色配方")
        res, res_all = predict_pure(target, match, match_df, match_columns, pure_colorants, lab, reflectance_coef,
                                    ignore_colorants, more_info)
    else:
        # print("颗粒配方")
        res, res_all = predict_particle(target, match, match_df, match_columns, pure_colorants, lab, reflectance_coef,
                                        particle_colorants, wl, formula, ignore_colorants, more_info)

    if res_all is None:
        return result

    best_count = len(res)
    res_merge = merge_colorants(res, layer_num == 3)
    res_other = [c for c in res if c not in set(res_merge)]
    res_all_new = [(colorant, delta) for (colorant, delta) in res_all if any(colorant == c for c, _ in res_merge)] + \
                  [(colorant, delta) for (colorant, delta) in res_all if any(colorant == c for c, _ in res_other)] + \
                  [(colorant, delta) for (colorant, delta) in res_all if not any(colorant == c for c, _ in res)]
    res_new = [(item[0][:-1] + item[0][-1].lower() if item[0].endswith('_C') or item[0].endswith('_P') else item[0],
                "all" if index < best_count and abs(item[1]) < 0.05
                else "unchange" if abs(item[1]) < 0.05
                else "add" if item[1] > 0 else "reduce")
               for index, item in enumerate(res_all_new)]
    result['order'] = dict(res_new)
    result['num'] = best_count
    # print(result)
    # end_time = time.time()
    # print("调色策略整体耗时：", end_time - start_time)
    return result

# app = FastAPI()
#
#
# @app.post("/color_strategy_two_new")
# async def toning_main(item: dict):
#     logging.info("开始加载模型")
#     try:
#         response = run(item)
#         return response
#     except Exception as e:
#         error_message = f"An error occurred: {str(e)}"
#         custom_detail = {"data": [], "detail": error_message}
#         return JSONResponse(status_code=400, content=custom_detail)
#
#
# def start():
#     uvicorn.run(app, host="192.168.100.128", port=9224)
#
#
# if __name__ == '__main__':
#     request = {"data":[{"colorPanel":{"angles":[{"cv":2.739904,"reflectance":[],"sa":3.91E-4,"sg":0.149151,"si":0.362773,"simple_coarseness":0.083753},{"cv":5.629368,"reflectance":[3.5790012,3.8499599,4.0439224,4.309297,4.5567665,4.975674,6.2780695,9.941558,17.03109,25.359627,31.545158,36.267384,41.7499,48.349022,54.061817,57.382305,58.712788,59.141968,59.309887,59.34626,59.26684,59.134197,58.992065,58.89787,58.732224,58.547646,58.31281,58.01219,57.80406,57.76912,57.699745],"sa":7.81E-4,"sg":0.427623,"si":0.236035,"simple_coarseness":0.125953},{"cv":1.592299,"reflectance":[3.043563,3.3057456,3.5037425,3.82895,4.0989575,4.406291,5.6922,9.317117,16.384373,24.773293,31.050365,35.86203,41.415997,48.118652,53.978622,57.368225,58.693897,59.147633,59.304913,59.345863,59.272636,59.127953,59.015457,58.905922,58.713276,58.521683,58.30129,58.01986,57.802307,57.77748,57.69369],"sa":0.0,"sg":0.0,"si":0.0,"simple_coarseness":0.064985},{"cv":2.487042,"reflectance":[3.7670062,4.094015,4.240647,4.4600096,4.659694,5.0079174,6.311907,10.00878,16.91864,24.815477,30.545538,34.873417,39.936504,45.953526,51.22184,54.235287,55.463463,55.83291,55.943268,55.943954,55.880295,55.787754,55.617886,55.53407,55.334717,55.209435,55.04148,54.773746,54.5375,54.48597,54.57079],"sa":0.0,"sg":0.0,"si":0.0,"simple_coarseness":0.11412}]},"color_code":"B83","dc":0.0,"device":"XRITE","match_result":"[\"A0011019704\",\"A011CPS4U10247188CL\",\"A011CPS4U5258385CL\",\"A011CPS3U6638979CL\",\"A00417218\",\"A011CPS3U1774057CL\",\"A011CPS4U7837754CL\",\"A011CPS3U2933684CL\",\"A011CPS4U6859959CL\",\"A011CPS3U2014743MS\",\"A011CPS4U10265532MS\",\"A002SJ202312030241\",\"A0011081152\",\"A007ASK001127\",\"A0011028671\"]","paint_type":"BASF_W","priority":"L2","serialNo":"A011CPS4U5258385CL","sysTenant":"Enoch","vehicle_brand":"奥迪","vehicle_spec":"A4","year":2004}]}
#     run(request)
    # start()
