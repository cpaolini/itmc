def get_coordinates(input_format: int, output_format: int, coords: tuple) -> tuple:
    """
    For input_format and output_format:
        8005 = Westward-facing Camera
        8006 = Eastward-facing Camera
        8007 = Northward-facing Camera
        8008 = Southward-facing Camera
        0 = Global Coordinate System (Birdseye)
        1 = Map Coordinates (Google Earth)
    """
    X__8005 = [10.30843405,45.38756318,196.4096603,1338.170572,-57.40383628]
    Y__8005 = [585.276743,-16.00967953,28.00210801,3.6280057,160.1384002]
    X__8006 = [195.2607212,-220.131988,-534.43058,-1203.554633,1640.809129]
    Y__8006 = [866.107049,-38.61118797,-1616.247628,141.9939402,815.9801013]
    X__8007 = [-250.7906888,361.3265714,1711.578691,-800.0496691,331.1810391]
    Y__8007 = [25.14681561,855.081163,-110.0547794,-1669.285951,977.7475742]
    X__8008 = [-328.5627317,250.3756283,-1126.659671,94.66940112,1310.884189]
    Y__8008 = [28.01475901,851.2439099,14.70547596,-86.02636125,29.94651496]
    X_8005 = [-0.0000000000134683, 0.0000000285247, -0.0000235506, 0.00978878, -1.00207]
    Y_8005 = [-0.0000000648198977,0.000000735736382,0.000783991199,-0.000747270962,0.131582098]
    X_8006 = [0.0000000375226214,0.00000266575432,-0.00015221015,-0.00336152476,1.33433451]
    Y_8006 = [0.000000014275798,-0.000000500160587,-0.000660893546,0.000685250173,0.841960017]
    X_8007 = [-0.0000000423174057,0.000000344145271,0.000672973435,-0.00061542074,0.138726529]
    Y_8007 = [0.0000000717811636,0.00000240674856,-0.000179874276,-0.00309910329,1.31155569]
    X_8008 = [0.0000000480810518,-0.000000371226596,-0.000677441086,0.000542384369,0.891987386]
    Y_8008 = [-0.0000000902242638,-0.00000215659355,0.000143667237,0.00273064377,-0.0508829428]
    acceptable_formats = [0, 8005, 8006, 8007, 8008]
    x, y = coords
    if input_format not in acceptable_formats or output_format not in acceptable_formats or input_format == output_format:
        raise ValueError("An input was submitted to get_birdseye_coordinates that is not one of the valid options. input_format and output_format can each be 8005, 8006, 8007, 8008, or 0, and must be different from each other.")
    io = (input_format, output_format)
    if io == (0, 8005): # Westward-facing camera
        output_x = (X__8005[0] * (x**2)) + (X__8005[1] * (y**2)) + (X__8005[2] * (x)) + (X__8005[3] * (y)) + (X__8005[4])
        output_y = (Y__8005[0] * (x**2)) + (Y__8005[1] * (y**2)) + (Y__8005[2] * (x)) + (Y__8005[3] * (y)) + (Y__8005[4])
    elif io == (0, 8006): # Eastward-facing camera
        output_x = (X__8006[0] * (x**2)) + (X__8006[1] * (y**2)) + (X__8006[2] * (x)) + (X__8006[3] * (y)) + (X__8006[4])
        output_y = (Y__8006[0] * (x**2)) + (Y__8006[1] * (y**2)) + (Y__8006[2] * (x)) + (Y__8006[3] * (y)) + (Y__8006[4])
    elif io == (0, 8007): # Northward-facing camera
        output_x = (X__8007[0] * (x**2)) + (X__8007[1] * (y**2)) + (X__8007[2] * (x)) + (X__8007[3] * (y)) + (X__8007[4])
        output_y = (Y__8007[0] * (x**2)) + (Y__8007[1] * (y**2)) + (Y__8007[2] * (x)) + (Y__8007[3] * (y)) + (Y__8007[4])
    elif io == (0, 8008): # Southward-facing camera
        output_x = (X__8008[0] * (x**2)) + (X__8008[1] * (y**2)) + (X__8008[2] * (x)) + (X__8008[3] * (y)) + (X__8008[4])
        output_y = (Y__8008[0] * (x**2)) + (Y__8008[1] * (y**2)) + (Y__8008[2] * (x)) + (Y__8008[3] * (y)) + (Y__8008[4])
    elif io == (8005, 0):
        output_x = (X_8005[0] * (y**4)) + (X_8005[1] * (y**3)) + (X_8005[2] * (y**2)) + (X_8005[3] * (y)) + (X_8005[4])
        a_b = (0, -254) # left_crosswalk_intersection
        c_d = (1280, -224) # right_crosswalk_intersection
        y = -y
        y_bound_left = ((157 - a_b[0]) / (-153 - a_b[1])) * (y - a_b[1])
        print(y_bound_left)
        y_bound_right = (y + 1280 * ((-153 - c_d[1]) / (1137 - c_d[0])) - c_d[1]) * ((1137 - c_d[0]) / (-153 - c_d[1]))
        print(y_bound_right)
        output_y = (x - y_bound_left) / (y_bound_right - y_bound_left)
    elif io == (8006, 0):
        output_x = (X_8006[0] * (x**2)) + (X_8006[1] * (y**2)) + (X_8006[2] * (x)) + (X_8006[3] * (y)) + (X_8006[4])
        output_y = (Y_8006[0] * (x**2)) + (Y_8006[1] * (y**2)) + (Y_8006[2] * (x)) + (Y_8006[3] * (y)) + (Y_8006[4])
    elif io == (8007, 0):
        output_x = (X_8007[0] * (x**2)) + (X_8007[1] * (y**2)) + (X_8007[2] * (x)) + (X_8007[3] * (y)) + (X_8007[4])
        output_y = (Y_8007[0] * (x**2)) + (Y_8007[1] * (y**2)) + (Y_8007[2] * (x)) + (Y_8007[3] * (y)) + (Y_8007[4])
    elif io == (8008, 0):
        output_x = (X_8008[0] * (x**2)) + (X_8008[1] * (y**2)) + (X_8008[2] * (x)) + (X_8008[3] * (y)) + (X_8008[4])
        output_y = (Y_8008[0] * (x**2)) + (Y_8008[1] * (y**2)) + (Y_8008[2] * (x)) + (Y_8008[3] * (y)) + (Y_8008[4])
    elif io == (0, 1):
        output_x = (0.2 * (((1981-1022) * x) + 1022))
        output_y = (0.2 * (((596-1555) * y) + 1555))
    else:
        if input_format != 0 and output_format != 0:
            return get_coordinates(0, output_format, get_coordinates(input_format, 0, coords))
    return (output_x, output_y)

# for val in [8005, 8006, 8007, 8008]:
#     print(get_coordinates(0, val, (0.486, 0.541)))
#     print(get_coordinates(val, 0, get_coordinates(0, val, (0.486, 0.541))))

print(get_coordinates(8005, 0, (384, 288)))