import ast

import numpy as np
import pandas as pd
import toml
from sqlalchemy import create_engine, text

from config import default

config = toml.load(default.Config_Path)


##数据库连接时处理数据函数
def DataFrame_SWitch_FormulaDetail(data):
    n = len(data['STEP'].unique())
    if n == 1:
        num = data['GRAM_PARTS'].sum()
        dic = {}
        for i in range(len(data)):
            dic[data.iloc[i]['COLORANT_CODE']] = data.iloc[i]['GRAM_PARTS'] / num * 100
        string = str([dic])
        return string, False
    else:
        data_c = data[data['STEP'] == 'C']
        data_p = data[data['STEP'] == 'P']
        dic_c = {}
        dic_p = {}
        num_c = data_c['GRAM_PARTS'].sum()
        num_p = data_p['GRAM_PARTS'].sum()
        for i in range(len(data_c)):
            dic_c[data_c.iloc[i]['COLORANT_CODE'] + '_c'] = data_c.iloc[i]['GRAM_PARTS'] / num_c * 100
        for j in range(len(data_p)):
            dic_p[data_p.iloc[j]['COLORANT_CODE'] + "_p"] = data_p.iloc[j]['GRAM_PARTS'] / num_p * 100
        string = str([dic_c, dic_p])
        return string, True


##数据库连接时处理数据函数
def DataFrame_Switch_List(data, type):
    if len(data) == 0:
        return '[]'
    else:
        data = data.iloc[:, type:].iloc[0].tolist()
        num = 0
        for i in data:
            if i is None:
                num += 1
        if num > 0:
            return '[]'
        else:
            return str(data)


##数据库连接时处理数据函数
def DataFrame_Switch_FlagCode(data):
    if len(data) == 0:
        return np.nan
    else:
        return data.iloc[0]['FLAG_CODE']


def Get_Particle_Combine_All(data):
    if len(data) == 0:
        return []
    else:
        return data.iloc[0]['particle_combine_all'].split(',')


##连接数据库获取数据
def Get_SQL_Data(value, sql, params=None):
    engine = create_engine(value)
    with engine.connect() as conn:  # 使用上下文管理器自动关闭连接
        df = pd.read_sql_query(text(sql), conn, params=params)
    return df


def Get_match_SQL_Data(code, id_code):
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
    value = config['PROD']['enoch_algorithm_db_url']

    if id_code is None:
        sql = """
                select FORMULA_CODE, gauge_id ID, flag_code, dc, sgm15, sg15, sg45, sg80, cvm15, cv15, cv45, cv80,
                       sam15, sa15, sa45, sa80, sim15, si15, si45, si80, sdm15, sd15, sd45, sd80,
                       simple_coarsenessM15, simple_coarseness15, simple_coarseness45, simple_coarseness80,
                       background_coarsenessM15, background_coarseness15, background_coarseness45, background_coarseness80,
                       directional_coarsenessM15, directional_coarseness15, directional_coarseness45, directional_coarseness80,
                       Angle15L, Angle15A, Angle15B, Angle45L, Angle45A, Angle45B, Angle110L, Angle110A, Angle110B, 
                       reflectance15_6coef, reflectance45_6coef, reflectance110_6coef, particle_combine_gram_all
                from dws_formula_gauge_with_texture
                where FORMULA_CODE = '{}'
                """.format(code)
    else:
        sql = """
            select FORMULA_CODE, gauge_id ID, flag_code, dc, sgm15, sg15, sg45, sg80, cvm15, cv15, cv45, cv80,
                   sam15, sa15, sa45, sa80, sim15, si15, si45, si80, sdm15, sd15, sd45, sd80,
                   simple_coarsenessM15, simple_coarseness15, simple_coarseness45, simple_coarseness80,
                   background_coarsenessM15, background_coarseness15, background_coarseness45, background_coarseness80,
                   directional_coarsenessM15, directional_coarseness15, directional_coarseness45, directional_coarseness80,
                   Angle15L, Angle15A, Angle15B, Angle45L, Angle45A, Angle45B, Angle110L, Angle110A, Angle110B, 
                   reflectance15_6coef, reflectance45_6coef, reflectance110_6coef, particle_combine_gram_all
            from dws_formula_gauge_with_texture
            where FORMULA_CODE = '{}' and gauge_id = '{}'
            """.format(code, id_code)

    data_all = Get_SQL_Data(value, sql)
    if len(data_all) == 0:
        return pd.DataFrame(), False, []
    data_all = data_all.iloc[0:1]
    formula_dict = ast.literal_eval(data_all.loc[0, 'particle_combine_gram_all'])
    layer_three = any(key.endswith('_P') for key in formula_dict)
    if layer_three:
        formula_suffixes = ['_C', '_P']
        formula = [item + suffix for item in formula for suffix in formula_suffixes]

    new_columns = pd.DataFrame(0, index=data_all.index, columns=formula)
    data_all = pd.concat([data_all, new_columns], axis=1)
    if layer_three:
        for key, value in formula_dict.items():
            data_all.at[0, key] = value
    else:
        for key, value in formula_dict.items():
            data_all.at[0, key.rstrip('_C')] = value
    formula_keys = list(formula_dict.keys())
    return data_all, layer_three, formula_keys


def Get_match_dataset_SQL_Data_With_Code(codes, layer_three):
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
    if layer_three:
        formula_suffixes = ['_C', '_P']
        formula = [item + suffix for item in formula for suffix in formula_suffixes]
    value = config['PROD']['enoch_algorithm_db_url']
    if codes:
        if len(codes) == 1:
            codes_id_codes = "('{}')".format(codes[0])
        else:
            codes_id_codes = tuple(codes)
            codes_id_codes_str = ', '.join(["'{}'".format(code) for code in codes_id_codes])
            codes_id_codes = "({})".format(codes_id_codes_str)
        where_clause = f"WHERE FORMULA_CODE IN {codes_id_codes}"
    else:
        where_clause = "WHERE 1 = 0"

    sql = f"""  
        SELECT FORMULA_CODE, gauge_id ID, dc, sgm15, sg15, sg45, sg80, cvm15, cv15, cv45, cv80,  
               sam15, sa15, sa45, sa80, sim15, si15, si45, si80, sdm15, sd15, sd45, sd80,  
               simple_coarsenessM15, simple_coarseness15, simple_coarseness45, simple_coarseness80,  
               background_coarsenessM15, background_coarseness15, background_coarseness45, background_coarseness80,  
               directional_coarsenessM15, directional_coarseness15, directional_coarseness45, directional_coarseness80,  
               Angle15L, Angle15A, Angle15B, Angle45L, Angle45A, Angle45B, Angle110L, Angle110A, Angle110B,  
               reflectance15_6coef, reflectance45_6coef, reflectance110_6coef, particle_combine_gram_all  
        FROM dws_formula_gauge_with_texture  
        {where_clause}  
    """

    data_all = Get_SQL_Data(value, sql)

    def process_row(row, formula, layer_three):
        formula_detail = row['particle_combine_gram_all']
        try:
            formula_dict = ast.literal_eval(formula_detail)
        except (ValueError, SyntaxError):
            return pd.Series(), False

        # 异或运算判断两个bool是否一致
        if any(key.endswith('_P') for key in formula_dict) ^ layer_three:
            return pd.Series(), False

        new_cols = {}
        for col in formula:
            new_cols[col] = 0

        if layer_three:
            for key, value in formula_dict.items():
                new_cols[key] = value
        else:
            for key, value in formula_dict.items():
                new_cols[key.rstrip('_C')] = value

        return pd.Series(new_cols), True

    results = []
    valid_flags = []
    for index, row in data_all.iterrows():
        new_cols, is_valid = process_row(row, formula, layer_three)
        results.append(new_cols)
        valid_flags.append(is_valid)

    new_cols_df = pd.DataFrame(results, index=data_all.index)
    valid_rows = pd.Series(valid_flags, index=data_all.index)

    data_all = pd.concat([data_all, new_cols_df], axis=1)
    data_all = data_all[valid_rows]
    return data_all


def Get_match_dataset_SQL_Data_With_Code_ID(codes_list, layer_three):
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
    if layer_three:
        formula_suffixes = ['_C', '_P']
        formula = [item + suffix for item in formula for suffix in formula_suffixes]
    value = config['PROD']['enoch_algorithm_db_url']

    if codes_list:
        if len(codes_list) == 1:
            codes_id_codes = f"({codes_list[0]})"
        else:
            codes_id_codes = ', '.join(f"('{code}', {id_code})" for code, id_code in codes_list)
        where_clause = f"WHERE (FORMULA_CODE, gauge_id) IN ({codes_id_codes})"
    else:
        where_clause = "WHERE 1 = 0"

    sql = f"""  
        SELECT FORMULA_CODE, gauge_id ID, dc, sgm15, sg15, sg45, sg80, cvm15, cv15, cv45, cv80,  
               sam15, sa15, sa45, sa80, sim15, si15, si45, si80, sdm15, sd15, sd45, sd80,  
               simple_coarsenessM15, simple_coarseness15, simple_coarseness45, simple_coarseness80,  
               background_coarsenessM15, background_coarseness15, background_coarseness45, background_coarseness80,  
               directional_coarsenessM15, directional_coarseness15, directional_coarseness45, directional_coarseness80,  
               Angle15L, Angle15A, Angle15B, Angle45L, Angle45A, Angle45B, Angle110L, Angle110A, Angle110B,  
               reflectance15_6coef, reflectance45_6coef, reflectance110_6coef, particle_combine_gram_all  
        FROM dws_formula_gauge_with_texture  
        {where_clause}  
    """

    data_all = Get_SQL_Data(value, sql)

    def process_row(row, formula, layer_three):
        formula_detail = row['particle_combine_gram_all']
        try:
            formula_dict = ast.literal_eval(formula_detail)
        except (ValueError, SyntaxError):
            return pd.Series(), False

        # 异或运算判断两个bool是否一致
        if any(key.endswith('_P') for key in formula_dict) ^ layer_three:
            return pd.Series(), False

        new_cols = {}
        for col in formula:
            new_cols[col] = 0

        if layer_three:
            for key, value in formula_dict.items():
                new_cols[key] = value
        else:
            for key, value in formula_dict.items():
                new_cols[key.rstrip('_C')] = value

        return pd.Series(new_cols), True

    results = []
    valid_flags = []
    for index, row in data_all.iterrows():
        new_cols, is_valid = process_row(row, formula, layer_three)
        results.append(new_cols)
        valid_flags.append(is_valid)

    new_cols_df = pd.DataFrame(results, index=data_all.index)
    valid_rows = pd.Series(valid_flags, index=data_all.index)

    data_all = pd.concat([data_all, new_cols_df], axis=1)
    data_all = data_all[valid_rows]
    return data_all
