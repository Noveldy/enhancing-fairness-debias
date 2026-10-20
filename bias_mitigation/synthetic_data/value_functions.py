import numpy as np
from math import *
import random

# Basic math primitives
def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
def div(x, y): return x / y if y != 0 else np.nan
def log(x): return np.log(np.abs(x)) if x != 0 else np.nan
def sqrt(x): return np.sqrt(np.abs(x))
def sin(x): return np.sin(x)
def cos(x): return np.cos(x)

VALUE_FUNC_FACTORY_REGISTRY = {}

def register_value_func(name=None):
    def decorator(func):
        key = name or func.__name__
        VALUE_FUNC_FACTORY_REGISTRY[key] = func
        return func
    return decorator

@register_value_func("trend_noise")
def trend_noise_factory(params=None):
    """
    Creates a linear trend with noise.
    """
    slope = params.get("slope", 1.0)
    intercept = params.get("intercept", 0)
    return lambda i: intercept + slope * i + np.random.normal(0, 1)

# Attach param metadata (optional but recommended)
trend_noise_factory._params = ["slope", "intercept"]


@register_value_func("flat_random")
def flat_random_factory(params=None):
    """
    Uniform random values between min_val and max_val.
    """
    min_val = params.get("min_val", 0)
    max_val = params.get("max_val", 10)
    return lambda i: np.random.uniform(min_val, max_val)

flat_random_factory._params = ["min_val", "max_val"]


@register_value_func("oscillate")
def oscillate_factory(params=None):
    """
    Produces an oscillating value based on a sine wave.
    """
    return lambda i: 50 + 10 * np.sin(i)


@register_value_func("auto_CO_GT")
def auto_CO_GT_factory(params=None):
    """
    Generate a synthetic approximation of the CO(GT) time series using Fourier components.
    """
    return lambda i: -34.207524 + 26.540117 * np.sin(2 * np.pi * 0.000107 * i + 0.575905) + 20.189981 * np.sin(2 * np.pi * 0.000321 * i + 0.667505) + 18.668993 * np.sin(2 * np.pi * 0.003420 * i + 2.186228) + 18.592031 * np.sin(2 * np.pi * 0.001603 * i + 1.137176) + 18.162409 * np.sin(2 * np.pi * 0.002779 * i + -1.106402) + np.random.normal(0, 84)

auto_CO_GT_factory._params = []

@register_value_func("auto_PT08_S1_CO")
def auto_PT08_S1_CO_factory(params=None):
    """
    Generate a synthetic approximation of the PT08_S1_CO time series using Fourier components.
    """
    return lambda i: 1048.990061 + 100.814271 * np.sin(2 * np.pi * 0.041680 * i + 0.300538) + 96.488043 * np.sin(2 * np.pi * 0.000214 * i + -0.796698) + 81.816974 * np.sin(2 * np.pi * 0.083360 * i + -2.066573) + 77.132663 * np.sin(2 * np.pi * 0.006733 * i + 0.752967) + 68.732942 * np.sin(2 * np.pi * 0.002458 * i + -1.757554) + np.random.normal(0, 356)

auto_PT08_S1_CO_factory._params = []

@register_value_func("auto_NMHC_GT")
def auto_NMHC_GT_factory(params=None):
    """
    Generate a synthetic approximation of the NMHC_GT time series using Fourier components.
    """
    return lambda i: -159.090093 + 79.447934 * np.sin(2 * np.pi * 0.000107 * i + -0.424697) + 72.595965 * np.sin(2 * np.pi * 0.000214 * i + -0.848934) + 62.021265 * np.sin(2 * np.pi * 0.000321 * i + -1.271949) + 48.878645 * np.sin(2 * np.pi * 0.000427 * i + -1.692062) + 34.575675 * np.sin(2 * np.pi * 0.000534 * i + -2.104784)

auto_NMHC_GT_factory._params = []

@register_value_func("auto_C6H6_GT")
def auto_C6H6_GT_factory(params=None):
    """
    Generate a synthetic approximation of the C6H6_GT time series using Fourier components.
    """
    return lambda i: 1.865683 + 11.005496 * np.sin(2 * np.pi * 0.004489 * i + 1.756737) + 10.331947 * np.sin(2 * np.pi * 0.006733 * i + 0.793699) + 9.932563 * np.sin(2 * np.pi * 0.002992 * i + 1.850148) + 8.973031 * np.sin(2 * np.pi * 0.003527 * i + 0.101169) + 8.276810 * np.sin(2 * np.pi * 0.002244 * i + 2.278528) + np.random.normal(0, 44)

auto_C6H6_GT_factory._params = []

@register_value_func("auto_PT08_S2_NMHC")
def auto_PT08_S2_NMHC_factory(params=None):
    """
    Generate a synthetic approximation of the PT08_S2_NMHC time series using Fourier components.
    """
    return lambda i: 894.595276 + 155.804623 * np.sin(2 * np.pi * 0.041680 * i + 0.283966) + 109.219019 * np.sin(2 * np.pi * 0.083360 * i + -2.073243) + 84.474671 * np.sin(2 * np.pi * 0.000214 * i + -1.232914) + 66.976797 * np.sin(2 * np.pi * 0.006733 * i + 0.851074) + 63.375219 * np.sin(2 * np.pi * 0.002458 * i + -1.818390) + np.random.normal(0, 378)

auto_PT08_S2_NMHC_factory._params = []

@register_value_func("auto_NOx_GT")
def auto_NOx_GT_factory(params=None):
    """
    Generate a synthetic approximation of the NOx_GT time series using Fourier components.
    """
    return lambda i: 168.616971 + 158.999936 * np.sin(2 * np.pi * 0.000107 * i + 1.336001) + 93.981106 * np.sin(2 * np.pi * 0.041680 * i + 0.272570) + 76.225516 * np.sin(2 * np.pi * 0.083360 * i + -2.389222) + 68.075288 * np.sin(2 * np.pi * 0.000321 * i + 0.595854) + 57.976241 * np.sin(2 * np.pi * 0.000214 * i + 3.111351) + np.random.normal(0, 300)

auto_NOx_GT_factory._params = []

@register_value_func("auto_PT08_S3_NOx")
def auto_PT08_S3_NOx_factory(params=None):
    """
    Generate a synthetic approximation of the PT08_S3_NOx time series using Fourier components.
    """
    return lambda i: 794.990168 + 123.982305 * np.sin(2 * np.pi * 0.041680 * i + -3.009878) + 92.633553 * np.sin(2 * np.pi * 0.000107 * i + -1.032119) + 92.023420 * np.sin(2 * np.pi * 0.083360 * i + 0.820634) + 75.140527 * np.sin(2 * np.pi * 0.005985 * i + 1.649271) + 68.453255 * np.sin(2 * np.pi * 0.001069 * i + -0.971697) + np.random.normal(0, 353)

auto_PT08_S3_NOx_factory._params = []

@register_value_func("auto_NO2_GT")
def auto_NO2_GT_factory(params=None):
    """
    Generate a synthetic approximation of the NO2_GT time series using Fourier components.
    """
    return lambda i: 58.148873 + 58.202574 * np.sin(2 * np.pi * 0.000107 * i + 0.616173) + 43.406465 * np.sin(2 * np.pi * 0.041680 * i + 0.182134) + 42.359505 * np.sin(2 * np.pi * 0.000321 * i + 0.627672) + 41.750208 * np.sin(2 * np.pi * 0.000214 * i + 2.474983) + 30.457590 * np.sin(2 * np.pi * 0.083360 * i + -2.404382) + np.random.normal(0, 144)

auto_NO2_GT_factory._params = []

@register_value_func("auto_PT08_S4_NO2")
def auto_PT08_S4_NO2_factory(params=None):
    """
    Generate a synthetic approximation of the PT08_S4_NO2 time series using Fourier components.
    """
    return lambda i: 1391.479641 + 293.663715 * np.sin(2 * np.pi * 0.000107 * i + -1.959513) + 150.240846 * np.sin(2 * np.pi * 0.000214 * i + -1.103735) + 110.404593 * np.sin(2 * np.pi * 0.083360 * i + -1.898735) + 105.534064 * np.sin(2 * np.pi * 0.041680 * i + 0.405809) + 92.597268 * np.sin(2 * np.pi * 0.006733 * i + 0.695521) + np.random.normal(0, 536)

auto_PT08_S4_NO2_factory._params = []

@register_value_func("auto_PT08_S5_O3")
def auto_PT08_S5_O3_factory(params=None):
    """
    Generate a synthetic approximation of the PT08_S5_O3 time series using Fourier components.
    """
    return lambda i: 975.072032 + 158.140776 * np.sin(2 * np.pi * 0.083360 * i + -2.213136) + 114.099979 * np.sin(2 * np.pi * 0.041680 * i + 0.225206) + 94.632851 * np.sin(2 * np.pi * 0.006733 * i + 0.882594) + 91.195201 * np.sin(2 * np.pi * 0.006092 * i + -2.323622) + 90.157869 * np.sin(2 * np.pi * 0.000641 * i + 0.642626) + np.random.normal(0, 490)

auto_PT08_S5_O3_factory._params = []

@register_value_func("auto_T")
def auto_T_factory(params=None):
    """
    Generate a synthetic approximation of the T time series using Fourier components.
    """
    return lambda i: 9.778305 + 14.413633 * np.sin(2 * np.pi * 0.000107 * i + -2.003383) + 11.237471 * np.sin(2 * np.pi * 0.004489 * i + 1.777907) + 10.709350 * np.sin(2 * np.pi * 0.006733 * i + 0.772819) + 10.487083 * np.sin(2 * np.pi * 0.002992 * i + 1.808063) + 9.547692 * np.sin(2 * np.pi * 0.003527 * i + 0.083374)

auto_T_factory._params = []

@register_value_func("auto_RH")
def auto_RH_factory(params=None):
    """
    Generate a synthetic approximation of the RH time series using Fourier components.
    """
    return lambda i: 39.485380 + 13.296126 * np.sin(2 * np.pi * 0.000214 * i + -0.890547) + 12.095798 * np.sin(2 * np.pi * 0.006733 * i + 0.760653) + 12.093909 * np.sin(2 * np.pi * 0.002992 * i + 1.743395) + 11.955103 * np.sin(2 * np.pi * 0.041680 * i + -2.951462) + 11.938912 * np.sin(2 * np.pi * 0.002137 * i + 0.990998)

auto_RH_factory._params = []

@register_value_func("auto_AH")
def auto_AH_factory(params=None):
    """
    Generate a synthetic approximation of the AH time series using Fourier components.
    """
    return lambda i: -6.837604 + 10.335090 * np.sin(2 * np.pi * 0.004489 * i + 1.784361) + 9.679325 * np.sin(2 * np.pi * 0.002992 * i + 1.822359) + 9.605967 * np.sin(2 * np.pi * 0.006733 * i + 0.784823) + 8.691842 * np.sin(2 * np.pi * 0.003527 * i + 0.067392) + 8.322840 * np.sin(2 * np.pi * 0.002244 * i + 2.261799)

auto_AH_factory._params = []
