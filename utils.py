import math
import numpy as np

# ==============================================================================
# MÓDULO: UTILITÁRIOS (Grids e Coordenadas)
# ==============================================================================

def autogrid(x, y, z, sx, sy, sz, buffer=0):
    """
    Cria um dicionário de definição de grid (malha) a partir do tamanho da célula
    e dos limites das amostras.
    
    Args:
        x, y, z (array-like): Coordenadas dos dados.
        sx, sy, sz (float): Dimensões do bloco nas direções X, Y, Z.
        buffer (float, optional): Margem extra ao redor dos pontos. Default é 0.
        
    Returns:
        dict: Dicionário contendo a definição do grid (ox, oy, oz, nx, ny, nz, sx, sy, sz).
    """
    # Garantir que sejam arrays do numpy para uso eficiente do .min() e .max()
    x = np.asarray(x)
    y = np.asarray(y)

    ox = x.min() - buffer
    oy = y.min() - buffer
    
    max_x = x.max() + buffer
    max_y = y.max() + buffer
    
    nx = math.ceil((max_x - ox) / sx)
    ny = math.ceil((max_y - oy) / sy)

    # Tratamento seguro para projetos 2D (onde Z é nulo ou matriz de zeros)
    if z is None or len(z) == 0 or (isinstance(z, np.ndarray) and np.all(z == 0)):
        oz = 0.0
        nz = 1
        sz = 1.0 if sz == 0 else sz # Evita divisão por zero
    else:
        z = np.asarray(z)
        oz = z.min() - buffer
        max_z = z.max() + buffer
        nz = math.ceil((max_z - oz) / sz)

    return {
        'ox': float(ox), 'oy': float(oy), 'oz': float(oz),
        'sx': float(sx), 'sy': float(sy), 'sz': float(sz),
        'nx': int(nx),   'ny': int(ny),   'nz': int(nz)
    }

def add_coord(grid):
    """
    Gera as coordenadas cartesianas (X, Y, Z) para todos os centros de bloco da malha.
    Totalmente vetorizado com NumPy para performance extrema (Substitui loops lentos).
    
    Args:
        grid (dict): Dicionário de definições do grid gerado pelo autogrid.
        
    Returns:
        np.ndarray: Matriz de coordenadas com formato (nx*ny*nz, 3)
    """
    # Gera os eixos 1D
    x_vals = grid['ox'] + np.arange(grid['nx']) * grid['sx']
    y_vals = grid['oy'] + np.arange(grid['ny']) * grid['sy']
    z_vals = grid['oz'] + np.arange(grid['nz']) * grid['sz']

    # O np.meshgrid com indexing='ij' na ordem (Z, Y, X) reproduz o comportamento 
    # exato do itertools.product original: o eixo X varia mais rápido, 
    # seguido por Y e finalmente Z (Padrão GSLIB).
    ZZ, YY, XX = np.meshgrid(z_vals, y_vals, x_vals, indexing='ij')

    # Empilha horizontalmente os vetores achatados (ravel) para criar a matriz [X, Y, Z]
    return np.column_stack((XX.ravel(), YY.ravel(), ZZ.ravel()))
