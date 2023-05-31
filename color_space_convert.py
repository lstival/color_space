from utils import *

def tensor_yuv_2_rgb(x, int_8=True):
    """
    Convert a tensor from YUV to RGB
    """

    min_y_u = -0.5
    max_y_u = 0.5

    try:
        y,u,v = torch.split(x, 1, dim=1)
    except:
        _, y,u,v = torch.split(x, 1, dim=1)

    yuv_to_rgb = K.color.YuvToRgb()

    y = scale_0_and_1(y)
    # u = scale_0_and_1(u)
    # v = scale_0_and_1(v)

    u = (u.clamp(-1, 1) + 1) / 2
    v = (v.clamp(-1, 1) + 1) / 2

    # u = scale_0_and_1(a)
    # v = scale_0_and_1(b)

    # u = (u * (max_y_u - min_y_u)) + min_y_u
    # v = (v * (max_y_u - min_y_u)) + min_y_u

    u = (u - 0.5)
    v = (v - 0.5)
    x = torch.cat([y, u, v], 1)

    x = yuv_to_rgb(x)
    if int_8:
        x = (scale_0_and_1(x)*255).type(torch.uint8)
    else:
        x = (scale_0_and_1(x))

    return x

def tensor_lab_2_rgb(x, int_8=True):
    """
    Convert a tensor from LAB to RGB
    First define the max and min values of LAb range
    Where L between [0,100] and a,b between [-128, 127]

    And return a image in RGB space with values between [0,1] if
    int_8 is False or between [0,255] if int_8 is True
    """

    min_a_b = -128
    max_a_b = 127

    lab_to_RGB = K.color.LabToRgb()

    try:
        l,a,b = torch.split(x, 1, dim=1)
    except:
        _, l,a,b = torch.split(x, 1, dim=1)

    l = scale_0_and_1(l) * 100
    a = scale_0_and_1(a)
    b = scale_0_and_1(b)

    a = (a * (max_a_b - min_a_b)) + min_a_b
    b = (b * (max_a_b - min_a_b)) + min_a_b

    x = torch.cat([l, a, b], 1)
    x = lab_to_RGB(x)

    if int_8:
        x = (scale_0_and_1(x)*255).type(torch.uint8)
    else:
        x = (scale_0_and_1(x))

    return x
    
def tensor_hlv_2_rgb(x, int_8=True):
    """
    Convert a tensor from LAB to RGB
    First define the max and min values of LAb range
    Where L between [0,100] and a,b between [-128, 127]

    And return a image in RGB space with values between [0,1] if
    int_8 is False or between [0,255] if int_8 is True
    """
    hsv_to_RGB = K.color.HsvToRgb()

    try:
        h,l,s = torch.split(x, 1, dim=1)
    except:
        _, h,l,s = torch.split(x, 1, dim=1)

    # h = scale_0_and_1(h)
    # l = scale_0_and_1(l)
    # s = scale_0_and_1(s)
    
    x = torch.cat([h, l, s], 1)
    x = hsv_to_RGB(x)

    if int_8:
        x = (scale_0_and_1(x)*255).type(torch.uint8)
    else:
        x = (scale_0_and_1(x))

    return x