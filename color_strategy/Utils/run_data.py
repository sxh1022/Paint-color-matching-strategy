import requests
import json
import pymysql
import pandas as pd
import warnings
import numpy as np
from config import default
warnings.filterwarnings('ignore')

def Get_Data_aliyun(sql):
    con = pymysql.connect(host="enoch-qa-master.mysql.polardb.rds.aliyuncs.com",
                          port=3306,
                          user="devrw",
                          password="!QAZ2wsx",
                          database="enoch_algorithm",
                          charset='utf8')
    table = pd.read_sql(sql, con)
    return table

def Get_Data_aliyun_online(sql):
    con = pymysql.connect(host="enoch-prod.rwlb.rds.aliyuncs.com",
                          port=3306,
                          user="quanjunjie",
                          password="!QAZ2wsx",
                          database="enoch_algorithm",
                          charset='utf8')
    table = pd.read_sql(sql, con)
    return table

def request_post(url, param):
    headers = {'content-type': 'application/json'}
    ret = requests.post(url, json=param, headers=headers, timeout=1000)
    text = json.loads(ret.text)
    return text

def Get_Lab_Wl_Data(code):
    sql = """
    select VEHICLE_BRAND, VEHICLE_SPEC, COLOR_CODE
    from enoch_algorithm.dws_formula
    where formula_code = '{}'
    """.format(code)
    data = Get_Data_aliyun(sql)
    if len(data) == 0:
        return None, None, None, None, []
    vehicle_brand = data.iloc[0]['VEHICLE_BRAND']
    vehicle_spec = data.iloc[0]['VEHICLE_SPEC']
    if pd.isnull(vehicle_brand):
        vehicle_brand = ""
    if pd.isnull(vehicle_spec):
        vehicle_spec = ""
    #     paint_type = data.iloc[0]['paint_type']
    #     year = int(data.iloc[0]['year'])
    color_code = data.iloc[0]['COLOR_CODE']

    sql = """
    select Angle15L, Angle15A, Angle15B, Angle45L, Angle45A, Angle45B, Angle110L, Angle110A, Angle110B,
    reflectance15, reflectance45, reflectance110
    from enoch_algorithm.dws_formula_gauge_with_texture 
    where formula_code = '{}'
    """.format(code)
    data = Get_Data_aliyun(sql)
    lab = data.iloc[0, :9].tolist()
    reflectance = data.iloc[0, 9:].tolist()

    sql = """
    select dc, sg15, sgM15, sg45, sg80,
    cv15, cvm15, cv45,
    cv80, simple_coarsenessM15, saM15, siM15, cvm15, sgM15,
    simple_coarseness15, sa15, si15, cv15, sg15,
    simple_coarseness45, sa45, si45, cv45, sg45,
    simple_coarseness80, sa80, si80, cv80, sg80
    from enoch_algorithm.dws_formula_gauge_with_texture 
    where formula_code = '{}'
    """.format(code)
    data = Get_Data_aliyun(sql)
    wl = data.iloc[0, :9].tolist()
    sparkle = [data.iloc[0, 9:14].tolist(), data.iloc[0, 14:19].tolist(), data.iloc[0, 19:24].tolist(),
               data.iloc[0, 24:29].tolist()]
    dc = data.iloc[0:, 0].values[0]
    return lab, reflectance, wl, sparkle, [dc, vehicle_brand, vehicle_spec, color_code]


##匹配得到3等级配方号
def Get_Match_Code(code):
    lab, reflectance, wl, sparkle, other = Get_Lab_Wl_Data(code)
    if lab is None:
        return [], []
    # 3.6
    # post_url = "http://47.97.115.166:18701/formula"
    # 3.5
    # post_url = "http://47.97.115.166:18007/formula"
    post_url = default.match_http
    request_param = {"data": [
        {
            "colorPanel": {
                "angles": [
                    {
                        "reflectance": [],
                        "simple_coarseness": sparkle[0][0],
                        "sa": sparkle[0][1],
                        "si": sparkle[0][2],
                        "cv": sparkle[0][3],
                        "sg": sparkle[0][4]
                    },
                    {
                        "reflectance": eval(reflectance[0]),
                        "simple_coarseness": sparkle[1][0],
                        "sa": sparkle[1][1],
                        "si": sparkle[1][2],
                        "cv": sparkle[1][3],
                        "sg": sparkle[1][4]
                    },
                    {
                        "reflectance": eval(reflectance[1]),
                        "simple_coarseness": sparkle[2][0],
                        "sa": sparkle[2][1],
                        "si": sparkle[2][2],
                        "cv": sparkle[2][3],
                        "sg": sparkle[2][4]
                    },
                    {
                        "reflectance": eval(reflectance[2]),
                        "simple_coarseness": sparkle[3][0],
                        "sa": sparkle[3][1],
                        "si": sparkle[3][2],
                        "cv": sparkle[3][3],
                        "sg": sparkle[3][4]
                    }
                ]
            },
            "dc": other[0],
            "vehicle_brand": other[1],
            "vehicle_spec": other[2],
            "paint_type": 'BASF_W',
            "color_code": other[3],
            "device": "XRITE",
            "sysTenant": "Enoch"
        }
    ]
    }
    res = request_post(post_url, request_param)
    if res is None:
        return [], []
    res = pd.DataFrame(res['data'])
    if len(res) == 0:
        return [], []
    res = res[res['priority'] != 'L3']
    result_list = [(row.formula_code, row.gauge_id) for row in res.itertuples(index=False)]
    return res['formula_code'].tolist(), result_list


##匹配得到3等级配方号
def Get_Match_Code_with_request_param(request_param):
    # print(request_param)
    post_url = default.match_http
    res = request_post(post_url, request_param)
    res = pd.DataFrame(res['data'])
    if len(res) == 0:
        return []
    res = res[res['priority'] != 'L3']
    return res['formula_code'].tolist()

def Get_Match_Code_with_request_param_id(request_param):
    if request_param is None:
        return [], []
    post_url = default.match_http
    # post_url = "http://47.97.115.166:18007/formula"
    res = request_post(post_url, request_param)
    res = pd.DataFrame(res['data'])
    if len(res) == 0:
        return [], []
    res = res[res['priority'] != 'L3']

    result_list = [(row.formula_code, row.gauge_id) for row in res.itertuples(index=False)]

    return res['formula_code'].tolist(), result_list
