import pandas as pd
import numpy as np
import numba
import math
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# MÓDULO: VARIOGRAFIA (MOTOR MATEMÁTICO PURO)
# ==============================================================================

@numba.njit
def _hdist(distancex, distancey, distancez):
    dist = np.zeros(distancex.shape[0], dtype=np.float64)
    for i in range(distancex.shape[0]):
        dist[i] = np.sqrt(distancex[i]**2 + distancey[i]**2 + distancez[i]**2) + 1e-10
    return dist

@numba.njit
def _xydist(distancex, distancey):
    dist = np.zeros(distancex.shape[0], dtype=np.float64)
    for i in range(distancex.shape[0]):
        dist[i] = np.sqrt(distancex[i]**2 + distancey[i]**2) + 1e-10
    return dist

@numba.njit
def _xdist(p1, p2):
    return p1[:, 0] - p2[:, 0]

@numba.njit
def _ydist(p1, p2):
    return p1[:, 1] - p2[:, 1]

@numba.njit
def _zdist(p1, p2):
    return p1[:, 2] - p2[:, 2]

@numba.njit
def _combin(points, max_dist):
    n = points.shape[0]
    max_possivel = (n * (n - 1)) // 2 
    
    out_p1 = np.zeros((max_possivel, 5), dtype=np.float64)
    out_p2 = np.zeros((max_possivel, 5), dtype=np.float64)
    
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            if dx < max_dist:
                dy = points[i, 1] - points[j, 1]
                if dy < max_dist:
                    dz = points[i, 2] - points[j, 2]
                    if dz < max_dist:
                        out_p1[count] = points[i]
                        out_p2[count] = points[j]
                        count += 1
                        
    return out_p1[:count], out_p2[:count]

def _distances(dataset, nlags, x_label, y_label, z_label, lagdistance, head_property, tail_property, choice):
    max_dist = (nlags + 1) * lagdistance 
    
    X = np.asarray(dataset[x_label].values, dtype=np.float64)
    Y = np.asarray(dataset[y_label].values, dtype=np.float64)
    Z = np.asarray(dataset[z_label].values, dtype=np.float64)
    HEAD = np.asarray(dataset[head_property].values, dtype=np.float64)
    TAIL = np.asarray(dataset[tail_property].values, dtype=np.float64)

    points = np.column_stack((X, Y, Z, HEAD, TAIL))

    if choice != 1.0:
        idx = np.random.randint(0, len(points), int(len(points)*choice))
        points = points[idx]

    p1, p2 = _combin(points, max_dist)
    
    if len(p1) == 0:
        return np.empty((0, 9), dtype=np.float64)

    distancex = _xdist(p1, p2) 
    distancey = _ydist(p1, p2) 
    distancez = _zdist(p1, p2)
    distanceh = _hdist(distancex, distancey, distancez)
    distancexy = _xydist(distancex, distancey)
    
    head_1 = p1[:, 3]
    head_2 = p1[:, 4]
    tail_1 = p2[:, 3]
    tail_2 = p2[:, 4]

    distance_dataframe = np.column_stack((distancex, distancey, distancez, distancexy, distanceh, head_1, head_2, tail_1, tail_2))
    return distance_dataframe[distance_dataframe[:, 4] < max_dist]

def __permissible_pairs(lag_multiply, lagdistance, lineartolerance, check_azimuth, check_dip, check_bandh, check_bandv, htol, vtol, hband, vband, omni, dist):
    minimum_range = lag_multiply*lagdistance - lineartolerance
    maximum_range = lag_multiply*lagdistance + lineartolerance

    if not omni:
        mask = (dist[:,4] >= minimum_range) & (dist[:,4] <= maximum_range) & (check_azimuth >= htol) & (check_dip >= vtol) & (check_bandh < hband) & (check_bandv < vband)
        filter_dist = dist[mask]
    else:
        mask = (dist[:,4] >= minimum_range) & (dist[:,4] <= maximum_range)
        filter_dist = dist[mask]
    return filter_dist

def __calculate_experimental(dist, lag_multiply, lagdistance, lineartolerance, check_azimuth, check_dip, check_bandh, check_bandv, htol, vtol, hband, vband, omni, type_var, flipped):
    points = __permissible_pairs(lag_multiply, lagdistance, lineartolerance, check_azimuth, check_dip, check_bandh, check_bandv, htol, vtol, hband, vband, omni, dist)
    
    if points.size != 0:
        number_of_pairs = float(points.shape[0])
        average_distance = points[:,4].mean()
        value_exp = 0
        
        # Índices -> 5:head_1, 6:head_2, 7:tail_1, 8:tail_2
        if type_var == 'Variogram': 
            value_exp = ((points[:,5] - points[:,7])*(points[:,6] - points[:,8]))/(2*number_of_pairs)
            value_exp = value_exp.sum()
            
        elif type_var == 'Covariogram': 
            cov = ((points[:,5] - points[:,5].mean())*(points[:,8]-points[:,8].mean())).sum() / number_of_pairs
            if flipped:
                # Covariograma invertido: Variância (C0) - Covariância (Ch)
                var_head = points[:,5].var()
                value_exp = var_head - cov
            else:
                value_exp = cov
                
        elif type_var == 'Correlogram':
            std_head = points[:,5].std()
            std_tail = points[:,8].std()
            if std_head > 0 and std_tail > 0:
                cov = ((points[:,5] - points[:,5].mean())*(points[:,8]-points[:,8].mean())).sum() / number_of_pairs
                rho = cov / (std_head * std_tail)
                value_exp = (1.0 - rho) if flipped else rho
            else:
                value_exp = np.nan
                
        elif type_var == 'Non_Ergodic_Correlogram':
            mean_head = points[:,5].mean()
            mean_tail = points[:,8].mean()
            std_head = points[:,5].std()
            std_tail = points[:,8].std()
            if std_head > 0 and std_tail > 0:
                cov_local = ((points[:,5] - mean_head) * (points[:,8] - mean_tail)).sum() / number_of_pairs
                rho_local = cov_local / (std_head * std_tail)
                value_exp = (1.0 - rho_local) if flipped else rho_local
            else:
                value_exp = np.nan
                
        elif type_var == 'PairWise':
            denominators = (points[:,6] + points[:,8])
            valid_idx = denominators != 0
            if valid_idx.sum() > 0:
                value_exp = 2*((points[valid_idx,5] - points[valid_idx,7])/denominators[valid_idx])**2 / number_of_pairs
                value_exp = value_exp.sum()
            else:
                value_exp = np.nan
                
        elif type_var == 'Relative_Variogram':
            average_tail = (points[:,7] + points[:,8])/2
            average_head = (points[:,5] + points[:,6])/2
            denom = (average_head + average_tail)**2
            valid_idx = denom != 0
            if valid_idx.sum() > 0:
                value_exp = 4*((points[valid_idx,5] - points[valid_idx,7])*(points[valid_idx,6] - points[valid_idx,8]))/(number_of_pairs*denom[valid_idx])
                value_exp = value_exp.sum()
            else:
                 value_exp = np.nan
                 
        return [value_exp, number_of_pairs, average_distance]
    return [np.nan , np.nan, np.nan]

def _calculate_experimental_function(dataset, x_label, y_label, z_label, type_var, lagdistance, lineartolerance, htol, vtol, hband, vband, azimuth, dip, nlags, head_property, tail_property, choice, omni, flipped):
    dist = _distances(dataset, nlags, x_label, y_label, z_label, lagdistance, head_property, tail_property, choice)
    
    if len(dist) == 0:
        return pd.DataFrame(columns=['Spatial continuity', 'Number of pairs', 'Average distance'])

    cos_Azimuth = np.cos(np.radians(90-azimuth))
    sin_Azimuth = np.sin(np.radians(90-azimuth))
    cos_Dip     = np.cos(np.radians(90-dip))
    sin_Dip     = np.sin(np.radians(90-dip))
    
    htol = np.abs(np.cos(np.radians(htol)))
    vtol= np.abs(np.cos(np.radians(vtol)))
    
    check_azimuth = np.abs((dist[:,0]*cos_Azimuth + dist[:,1]*sin_Azimuth)/(dist[:,3] + 1e-10))
    check_dip     = np.abs((dist[:,3]*sin_Dip + dist[:,2]*cos_Dip)/(dist[:,4] + 1e-10))
    check_bandh   = np.abs(cos_Azimuth*dist[:,1] - sin_Azimuth*dist[:,0])
    check_bandv   = np.abs(sin_Dip*dist[:,2] - cos_Dip*dist[:,3])

    number_of_int = range(1, (nlags +1))
    
    value_exp_list = []
    for i in number_of_int:
        res = __calculate_experimental(dist, i, lagdistance, lineartolerance, check_azimuth, check_dip, check_bandh, check_bandv, htol, vtol, hband, vband, omni, type_var, flipped)
        value_exp_list.append(res)
        
    value_exp = np.array(value_exp_list, dtype=np.float64)
    df = pd.DataFrame(value_exp, columns=['Spatial continuity', 'Number of pairs', 'Average distance']).dropna()
    return df

def experimental(dataset, X, Y, head, tail, type_c, nlags, lagsize, lineartol, htol, vtol, hband, vband, azimuth, dip, omni, flipped, Z=None, choice=1.0):
    """
    Calcula o variograma experimental (ou covariograma/correlograma).
    Parâmetro 'flipped' inverte as curvas de correlação para ajustarem em modelos teóricos.
    """
    ndirections = len(nlags)
    if Z is None:
        dataset['Z'] = np.zeros(dataset[X].values.shape[0])
        Z = 'Z'    

    dfs = []
    returning = {'Directions': [azimuth, dip], 'Values': dfs}

    for i in range(ndirections):
        # type_c, nlags, lagsize, etc., são passados como listas na chamada
        res_df = _calculate_experimental_function(dataset, X, Y, Z, type_c[i], lagsize[i], lineartol[i], htol[i], vtol[i], hband[i], vband[i], azimuth[i], dip[i], nlags[i], head, tail, choice, omni[i], flipped[i])
        dfs.append(res_df)
        
    return returning

def calcular_modelo_teorico(experimental_dataframe, azimuth, dip, rotation_reference, model_func, ranges, contribution, nugget, inverted=False):
    # ... (Matemática mantida exatamente a mesma do código anterior) ...
    # (Para não estender o bloco, esta função continua recebendo a rotação 3D normal)
    if len(ranges[0]) != 3:
        raise ValueError("O número de direções principais deve ser menor ou igual a 3")
    if len(ranges) != len(contribution) or len(ranges) != len(model_func):
        raise ValueError("As estruturas do variograma devem ter o mesmo tamanho")

    y = math.cos(math.radians(dip)) * math.cos(math.radians(azimuth))
    x = math.cos(math.radians(dip)) * math.sin(math.radians(azimuth))  
    z = math.sin(math.radians(dip))

    angle_azimuth, angle_dip, angle_rake = math.radians(rotation_reference[0]), math.radians(rotation_reference[1]), math.radians(rotation_reference[2])

    rotation1 = np.array([[math.cos(angle_azimuth), -math.sin(angle_azimuth), 0], [math.sin(angle_azimuth), math.cos(angle_azimuth), 0], [0, 0, 1]])
    rotation2 = np.array([[1, 0, 0], [0, math.cos(angle_dip), math.sin(angle_dip)], [0, -math.sin(angle_dip), math.cos(angle_dip)]])
    rotation3 = np.array([[math.cos(angle_rake), 0, -math.sin(angle_rake)], [0, 1, 0], [math.sin(angle_rake), 0, math.cos(angle_rake)]])

    rot1 = np.dot(rotation1.T, np.array([x, y, z]))
    rot2 = np.dot(rotation2.T, rot1)
    rot3 = np.dot(rotation3.T, rot2)

    rotated_range = []
    for i in ranges:
        rotated = np.multiply(rot3, np.array([float(i[1]), float(i[0]), float(i[2])]).T)
        rotated_range.append(math.sqrt(rotated[0]**2 + rotated[1]**2 + rotated[2]**2))

    distancemax = experimental_dataframe['Average distance'].max()
    distancemax = distancemax if not math.isnan(distancemax) else 100.0
    distances = np.linspace(0.001, distancemax * 1.1, 200)

    model = []
    if not inverted:
        for i in distances:
            soma = 0
            for j, o, l in zip(contribution, model_func, rotated_range):
                if o == 'Exponential': soma += j * (1 - math.exp(-3 * i / float(l)))
                elif o == 'Gaussian': soma += j * (1 - math.exp(-3 * (i / float(l))**2))
                elif o == 'Spherical': soma += j * (1.5 * i / float(l) - 0.5 * (i / float(l))**3) if i <= l else j
            model.append(soma + nugget)
    else:
        for i in distances:
            soma = 0
            for j, o, l in zip(contribution, model_func, rotated_range):
                if o == 'Exponential': soma += (j + nugget) * (math.exp(-3 * i / float(l)))
                elif o == 'Gaussian': soma += (j + nugget) * (math.exp(-3 * (i / float(l)**2)))
                elif o == 'Spherical': soma += (j + nugget) * (1 - (1.5 * i / float(l) - 0.5 * (i / float(l))**3)) if i <= l else 0
            model.append(soma)

    return pd.DataFrame(np.array([distances, model]).T, columns=['distances', 'model'])


def calcular_hscatter_data(dataset, X, Y, head, tail, nlags, lagsize, lineartol, htol, vtol, hband, vband, azimuth, dip, omni, Z=None, choice=1.0):
    """
    Retorna os pares (Head, Tail) para cada Lag, permitindo a plotagem do H-Scatterplot.
    """
    if Z is None:
        dataset_copy = dataset.copy()
        dataset_copy['Z'] = np.zeros(dataset_copy[X].values.shape[0])
        Z = 'Z'
    else:
        dataset_copy = dataset

    # Calcula todas as distâncias usando o motor otimizado
    dist = _distances(dataset_copy, nlags, X, Y, Z, lagsize, head, tail, choice)
    
    if len(dist) == 0:
        return {}

    cos_Azimuth = np.cos(np.radians(90-azimuth))
    sin_Azimuth = np.sin(np.radians(90-azimuth))
    cos_Dip     = np.cos(np.radians(90-dip))
    sin_Dip     = np.sin(np.radians(90-dip))
    
    htol_rad = np.abs(np.cos(np.radians(htol)))
    vtol_rad = np.abs(np.cos(np.radians(vtol)))
    
    check_azimuth = np.abs((dist[:,0]*cos_Azimuth + dist[:,1]*sin_Azimuth)/(dist[:,3] + 1e-10))
    check_dip     = np.abs((dist[:,3]*sin_Dip + dist[:,2]*cos_Dip)/(dist[:,4] + 1e-10))
    check_bandh   = np.abs(cos_Azimuth*dist[:,1] - sin_Azimuth*dist[:,0])
    check_bandv   = np.abs(sin_Dip*dist[:,2] - cos_Dip*dist[:,3])

    hscatter_data = {}
    
    for i in range(1, nlags + 1):
        points = __permissible_pairs(i, lagsize, lineartol, check_azimuth, check_dip, check_bandh, check_bandv, htol_rad, vtol_rad, hband, vband, omni, dist)
        
        if points.size != 0:
            # Pegamos o valor exato da Cabeça e da Cauda para o par
            # 5: head_1, 6: head_2, 7: tail_1, 8: tail_2
            # Tradicionalmente, plota-se a média do ponto 1 vs média do ponto 2
            val_head = (points[:,5] + points[:,6]) / 2.0
            val_tail = (points[:,7] + points[:,8]) / 2.0
            
            hscatter_data[f'Lag_{i}'] = {
                'head': val_head,
                'tail': val_tail,
                'dist_media': points[:,4].mean(),
                'n_pares': len(val_head)
            }
            
    return hscatter_data

def calcular_varmap_data(dataset, X, Y, head, tail, type_c, nlags, lagsize, lineartol, htol, vtol, hband, vband, dip=0.0, step_angulo=15, Z=None, choice=1.0, flipped=False):
    """
    Roda a variografia em 360 graus e converte para coordenadas cartesianas X,Y
    preparando os dados para plotagem de um mapa de contorno (Varmap).
    """
    if Z is None:
        dataset_copy = dataset.copy()
        dataset_copy['Z'] = np.zeros(dataset_copy[X].values.shape[0])
        Z = 'Z'
    else:
        dataset_copy = dataset
        
    x_coords = []
    y_coords = []
    z_values = [] # Valores de continuidade espacial
    
    # Roda de 0 até 360 graus, saltando conforme o step (ex: de 15 em 15 graus)
    for azm in range(0, 360, step_angulo):
        df_exp = _calculate_experimental_function(
            dataset_copy, X, Y, Z, type_c, lagsize, lineartol, htol, vtol, 
            hband, vband, azm, dip, nlags, head, tail, choice, omni=False, flipped=flipped
        )
        
        if not df_exp.empty:
            lags_dist = df_exp['Average distance'].values
            valores = df_exp['Spatial continuity'].values
            
            # Converte a coordenada Polar (Distância, Ângulo) para Cartesiana (X, Y)
            # Obs: Multiplicamos por sin e cos de acordo com a regra de azimute norte (0)
            azm_rad = np.radians(azm)
            x_calc = lags_dist * np.sin(azm_rad)
            y_calc = lags_dist * np.cos(azm_rad)
            
            x_coords.extend(x_calc)
            y_coords.extend(y_calc)
            z_values.extend(valores)
            
    # Adiciona o ponto (0,0) com valor de variância 0 para o centro do alvo
    x_coords.append(0.0)
    y_coords.append(0.0)
    z_values.append(0.0 if not flipped else 1.0) # Se for correlograma flipped, centro é 1.0
    
    df_varmap = pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'valor': z_values
    })
    
    return df_varmap
