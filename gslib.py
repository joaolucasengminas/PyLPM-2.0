import pandas as pd
import numpy as np

# ==============================================================================
# MÓDULO: GSLIB I/O (Leitura e Escrita de Arquivos GeoEAS / GSLIB)
# ==============================================================================

def read_gslib(filepath, nodata=-999.0):
    """
    Lê um arquivo de dados no formato clássico GSLIB (GeoEAS) 
    e o converte em um Pandas DataFrame.
    
    Args:
        filepath (str): Caminho do arquivo a ser lido.
        nodata (float, optional): Valor numérico usado para representar nulos (NaN). 
                                  O padrão do GSLIB é -999.0 ou -99.0.
                                  
    Returns:
        pd.DataFrame: DataFrame contendo as variáveis e dados.
    """
    with open(filepath, 'r') as f:
        # A primeira linha é o título/nome do projeto
        title = f.readline().strip()
        
        # A segunda linha contém o número de variáveis
        n_vars_line = f.readline().strip().split()
        if not n_vars_line:
            raise ValueError(f"Arquivo '{filepath}' parece estar vazio ou fora do formato GSLIB.")
            
        n_vars = int(n_vars_line[0])
        
        # As próximas 'n_vars' linhas contêm os nomes das variáveis
        col_names = []
        for _ in range(n_vars):
            line = f.readline().strip().split()
            # Pega o primeiro elemento da linha como nome da coluna
            col_names.append(line[0] if line else f"Var_{len(col_names)+1}")
            
    # O número de linhas do cabeçalho que o Pandas deve pular
    skip_lines = 2 + n_vars
    
    # Lê os dados puramente numéricos usando o delimitador de espaços em branco
    df = pd.read_csv(filepath, delim_whitespace=True, skiprows=skip_lines, names=col_names)
    
    # Substitui os valores de 'nodata' por NaN do NumPy para cálculos seguros
    if nodata is not None:
        df.replace(nodata, np.nan, inplace=True)
        
    # Salva o título do arquivo como um atributo do DataFrame para registro
    df.attrs['title'] = title
    
    return df


def write_gslib(df, filepath, title="Data exported from Python", nodata=-999.0):
    """
    Escreve um Pandas DataFrame para o formato clássico GSLIB (GeoEAS).
    
    Args:
        df (pd.DataFrame): O DataFrame contendo os dados.
        filepath (str): O caminho e nome do arquivo de saída (ex: 'out.dat').
        title (str, optional): Título a ser escrito na primeira linha.
        nodata (float, optional): Valor numérico que substituirá os NaNs no texto.
    """
    # Cria uma cópia para não alterar o dataframe original da memória
    out_df = df.copy()
    
    # Substitui NaN/Nulls pelo valor de nodata oficial (-999.0)
    out_df.fillna(nodata, inplace=True)
    
    columns = out_df.columns.tolist()
    n_vars = len(columns)
    
    with open(filepath, 'w') as f:
        # Linha 1: Título
        f.write(f"{title}\n")
        # Linha 2: Número de Variáveis
        f.write(f"{n_vars}\n")
        # Linha 3 a 3+N: Nome das Variáveis
        for col in columns:
            f.write(f"{col}\n")
            
        # Escreve a matriz de dados
        # O to_string converte os dados garantindo um bom alinhamento em colunas textuais
        values = out_df.to_string(header=False, index=False)
        f.write(f"{values}\n")
