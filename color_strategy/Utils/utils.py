import math
from colour import (MSDS_CMFS, SDS_ILLUMINANTS,
                    SpectralDistribution, SpectralShape, CCS_ILLUMINANTS, Lab_to_LCHab, XYZ_to_RGB, Lab_to_XYZ)
import numpy as np
import pandas as pd
import colour
import ast


def get_coef(ref):
    if isinstance(ref, str):
        Y = ast.literal_eval(ref)
    else:
        Y = ref

    # X = np.linspace(4, 7, 31)
    X = np.linspace(400, 700, 31)
    coef = np.polyfit(X, Y, 5)

    Y_poly = np.polyval(coef, X)
    return coef.tolist()
    # return Y_poly.tolist()


def delta_wl(target_wl, match_wl):
    # if len(target_wl) == 0:
    #     return [0, 0, 0]
    ##纹理指标评价函数
    ##一个dc值、四个角度的sg值、四个角度的cv值
    dc = abs(target_wl[0] - match_wl[0])
    sg_cmc = ((abs(target_wl[1] - match_wl[1]) ** 2 + abs(target_wl[2] - match_wl[2]) ** 2 +
               abs(target_wl[3] - match_wl[3]) ** 2 + abs(target_wl[4] - match_wl[4]) ** 2) / 4) ** (1 / 2)
    cv_cmc = ((abs(target_wl[5] - match_wl[5]) ** 2 + abs(target_wl[6] - match_wl[6]) ** 2 +
               abs(target_wl[7] - match_wl[7]) ** 2 + abs(target_wl[8] - match_wl[8]) ** 2) / 4) ** (1 / 2)
    return [dc, sg_cmc, cv_cmc]


##BYK色差值
def delta_e(li_tv, li_nf, flag_code='S'):
    if flag_code == 'S':
        value_15, value_45, value_110 = 45, 45, 45
    else:
        value_15, value_45, value_110 = 15, 45, 110
    del15 = del_E(li_tv[:3], li_nf[:3], value_15)
    del45 = del_E(li_tv[3:6], li_nf[3:6], value_45)
    del110 = del_E(li_tv[6:], li_nf[6:], value_110)
    del_total = E_total([del15, del45, del110])
    return del_total


def delta_e_3angle(li_tv, li_nf, flag_code='S'):
    if flag_code == 'S':
        value_15, value_45, value_110 = 45, 45, 45
    else:
        value_15, value_45, value_110 = 15, 45, 110
    del15 = del_E(li_tv[:3], li_nf[:3], value_15)
    del45 = del_E(li_tv[3:6], li_nf[3:6], value_45)
    del110 = del_E(li_tv[6:], li_nf[6:], value_110)
    del_total = E_total([del15, del45, del110])
    return del15, del45, del110, del_total


def delta_e_3angle_list(li_tv, li_nf, flag_code='S'):
    if flag_code == 'S':
        value_15, value_45, value_110 = 45, 45, 45
    else:
        value_15, value_45, value_110 = 15, 45, 110
    del15 = del_E(li_tv[:3], li_nf[:3], value_15)
    del45 = del_E(li_tv[3:6], li_nf[3:6], value_45)
    del110 = del_E(li_tv[6:], li_nf[6:], value_110)
    return [del15, del45, del110]


def del_H(c1, h1, c2, h2):
    dh = h1 - h2
    if dh > 180:
        dh -= 360
    elif dh < -180:
        dh += 360
    return 2 * math.sqrt(c1 * c2) * math.sin(math.radians(dh / 2))


def del_E(ST, SP, angle):
    """
    BYK 单角度deltaE计算
    :param ST: 具体角度光谱1下的对应LAB值
    :param SP: 具体角度光谱2下的对应LAB值
    :param angle: 计算del_E的光谱角度
    :return:
    """
    L1, A1, B1 = ST
    L2, A2, B2 = SP

    _, C1, H1 = colour.Lab_to_LCHab(ST)
    _, C2, H2 = colour.Lab_to_LCHab(SP)
    dl = L1 - L2
    da = A1 - A2
    db = B1 - B2
    dc = C1 - C2
    dH = del_H(C1, H1, C2, H2)
    L = (L1 * L2) ** 0.5
    C = (C1 * C2) ** 0.5
    Sl = 0.15 * (L ** 0.5) + 31.5 / angle
    Sc = max(0.7, 0.48 * (C ** 0.5) - 0.35 * (L ** 0.5) + 42 / angle)
    Sh = max(0.7, 0.14 * (C ** 0.5) - 0.2 * (L ** 0.5) + 21 / angle + 0.7)
    Sa = 0.7
    Sb = 0.7
    del_ab = ((dl / (2 * Sl)) ** 2 + (da / (1.2 * Sa)) ** 2 + (db / (1.2 * Sb)) ** 2) ** 0.5
    del_ch = ((dl / (2 * Sl)) ** 2 + (dc / (1.8 * Sc)) ** 2 + (dH / (1.2 * Sh)) ** 2) ** 0.5
    Co = 10 + 8 / (1 + np.exp(27 - L))

    Co = Co.real if isinstance(Co, complex) else Co
    seta = 1 / (1 + np.exp(C - Co))

    seta = seta.real if isinstance(seta, complex) else seta
    del_e = seta * del_ab + (1 - seta) * del_ch
    del_e = del_e.real if isinstance(del_e, complex) else del_e
    return del_e


def E_total(E_list):
    return (np.sum([i ** 2 for i in E_list]) / len(E_list)) ** 0.5


def Lab_to_Rgb(lab):
    """
    LAB 转换为 RGB
    :param lab:
    :return:
    """
    y = (lab[0] + 16.0) / 116.0
    x = lab[1] / 500.0 + y
    z = y - lab[2] / 200.0

    y = y ** 3 if y > 6.0 / 29 else (y - 16.0 / 116) / 7.787
    x = x ** 3 if x > 6.0 / 29 else (x - 16.0 / 116) / 7.787
    z = z ** 3 if z > 6.0 / 29 else (z - 16.0 / 116) / 7.787

    x *= 0.95047
    z *= 1.08883

    rgb_r = 3.2406 * x - 1.5372 * y - 0.4986 * z
    rgb_g = -0.9689 * x + 1.8758 * y + 0.0415 * z
    rgb_b = 0.0557 * x - 0.2040 * y + 1.0570 * z

    rgb_r = 1.055 * rgb_r ** (1 / 2.4) - 0.055 if rgb_r > 0.0031308 else 12.92 * rgb_r
    rgb_g = 1.055 * rgb_g ** (1 / 2.4) - 0.055 if rgb_g > 0.0031308 else 12.92 * rgb_g
    rgb_b = 1.055 * rgb_b ** (1 / 2.4) - 0.055 if rgb_b > 0.0031308 else 12.92 * rgb_b

    return [min(max(int(rgb_r * 255), 0), 255), min(max(int(rgb_g * 255), 0), 255), min(max(int(rgb_b * 255), 0), 255)]


def Refl_2_Lab(refl):
    """
        对光谱数据进行转换lab等值操作
        :param refl: 光谱值，以,切割
        :param spectrum: 是否采用平滑处理
        :return: :class:`list`
        *CIE L\\*a\\*b\\** colourspace array.
        """
    cmfs = MSDS_CMFS["CIE 1964 10 Degree Standard Observer"]
    illuminant = SDS_ILLUMINANTS["D65"]
    shape = SpectralShape(400, 700, 10)
    data = np.array(refl)
    sd = SpectralDistribution(np.array([i / 100 for i in data]), shape)
    xyz = colour.sd_to_XYZ(sd, cmfs, illuminant).tolist()
    lab = colour.XYZ_to_Lab(np.array([i / 100 for i in xyz]), CCS_ILLUMINANTS[
        "CIE 1964 10 Degree Standard Observer"
    ]["D65"]).tolist()
    return lab


def rgb_to_hsv(red, green, blue):
    """
    根据rgb，hsv转换规则计算
    :param red:
    :param green:
    :param blue:
    :return:
    """
    # Normalize RGB values to range 0-1
    red /= 255.0
    green /= 255.0
    blue /= 255.0

    max_value = max(red, green, blue)
    min_value = min(red, green, blue)
    hue = 0
    saturation = 0
    value = max_value

    if max_value != 0 and max_value != min_value:
        saturation = (max_value - min_value) / max_value

        if max_value == red:
            hue = (green - blue) / (max_value - min_value) * 60
        elif max_value == green:
            hue = 120 + (blue - red) / (max_value - min_value) * 60
        else:
            hue = 240 + (red - green) / (max_value - min_value) * 60

        if hue < 0:
            hue += 360
    else:
        hue = 0
        saturation = 0

    # Scale HSV values to range 0-255, compare with 0 and 255
    hue = min(max(int(hue / 2), 0), 180)
    saturation = min(max(int(saturation * 255), 0), 255)
    value = min(max(int(value * 255), 0), 255)

    return hue, saturation, value


def color_dict():
    color_colorant_dict = {
        'black': ['90-A927', '90-A924', '90-A926', '90-A997', '98-M930', '90-1250'],
        'white': ['98-M919', '98-A097', '11-E-014', '90-M99/02', '93-M011', '90-905', '93-M010', '90-M99/24',
                  '90-A032', '90-M99/00', '11-E-025', '90-A035', '90-M99/01', '11-E-435', '90-M99/23', '90-M99/03',
                  '90-M99/04', '90-A031'],
        'green': ['90-A695', '11-E-650', '11-E-620', '11-E-680', '11-E-630', '90-A640'],
        'red': ['90-A350', '90-A378', '98-M319', '90-A306', '90-3A0', '11-E-330', '93-M364', '90-A359', '90-A323',
                '90-A307', '90-A372', '93-M363', '90-A347'],
        'blue': ['90-A528', '93-M506', '11-E-520', '90-A589', '90-A563', '90-A503', '11-E-660', '93-M505',
                 '11-E-480', '90-A527'],
        'yellow': ['11-E-120', '11-E-850', '90-A143', '11-E-910', '90-A177', '11-E-830', '90-A148', '90-A115',
                   '90-A105', '11-E-920', '93-M176', '90-A136', '90-A155', '90-A149'],
        'orange': ['90-A201', '11-E-220', '11-E-280', '90-A329'],
        'purple': ['90-A427', '11-E-460', '90-A430', '11-E-440'],
    }
    colorant_dict = {
        'M': ['90-905', '90-M99/00', '90-M99/01', '90-M99/02', '90-M99/03', '90-M99/04', '90-M99/23', '90-M99/24'],
        'black': ['90-A927', '90-A924', '90-A926', '90-A997', '98-M930', '90-1250'],
        'white': ['98-M919', '98-A097', '11-E-014', '93-M011', '93-M010',
                  '90-A032', '11-E-025', '90-A035', '11-E-435', '90-A031'],
        'green': ['90-A695', '11-E-650', '11-E-620', '11-E-680', '11-E-630', '90-A640'],
        'red': ['90-A350', '90-A378', '98-M319', '90-A306', '90-3A0', '11-E-330', '93-M364', '90-A359', '90-A323',
                '90-A307', '90-A372', '93-M363', '90-A347'],
        'blue': ['90-A528', '93-M506', '11-E-520', '90-A589', '90-A563', '90-A503', '11-E-660', '93-M505',
                 '11-E-480', '90-A527'],
        'yellow': ['11-E-120', '11-E-850', '90-A143', '11-E-910', '90-A177', '11-E-830', '90-A148', '90-A115',
                   '90-A105', '11-E-920', '93-M176', '90-A136', '90-A155', '90-A149'],
        'orange': ['90-A201', '11-E-220', '11-E-280', '90-A329'],
        'purple': ['90-A427', '11-E-460', '90-A430', '11-E-440'],
    }
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
