import numpy as np
import pandas as pd
import math
import warnings
from scipy import stats
from scipy.stats import gaussian_kde

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import griddata

matplotlib.use('Agg') # Evita que o Matplotlib tente abrir janelas soltas

# IMPORTA O SEU MÓDULO UTILS OTIMIZADO
from . import utils
warnings.filterwarnings('ignore')

# ==============================================================================
# FUNÇÕES AUXILIARES MATEMÁTICAS
# ==============================================================================

def weighted_avg_and_var(values, weights):
    """Calcula média e variância ponderadas."""
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return average, variance

def isotopic_arrays(arrays):
    """Extrai grupos isotópicos de arrays (remove NaNs de forma pareada)."""
    masks = [np.isnan(array) for array in arrays]
    mask = sum(masks)
    masked_arrays = [np.ma.array(array, mask=mask).compressed() for array in arrays]
    return masked_arrays

# ==============================================================================
# FÁBRICA DE GRÁFICOS (Retornam objetos go.Figure para o Panel)
# ==============================================================================

def plot_experimental_variogram(dfs, azm, dip, show_lines=True):
    """Plota uma grade de variogramas experimentais."""
    size_row = 1 if len(azm) < 4 else int(math.ceil(len(azm)/4))
    size_cols = 4 if len(azm) >= 4 else int(len(azm))

    titles = [f"Azimuth {azm[j]} - Dip {dip[j]}" for j in range(len(azm))]
    fig = make_subplots(rows=size_row, cols=size_cols, subplot_titles=titles)

    count_row, count_cols = 1, 1
    
    # Define se plota linhas + marcadores + texto, ou apenas marcadores + texto
    plot_mode = 'markers+lines+text' if show_lines else 'markers+text'

    for i in dfs:
        fig.add_trace(go.Scatter(
            x=i['Average distance'], y=i['Spatial continuity'],
            mode=plot_mode, 
            name='Experimental',
            marker=dict(color='#1f77b4', size=8), # Cor sólida e sem atributo colorscale
            text=i['Number of pairs'], 
            textposition='top center', # Posiciona o número acima do ponto
            textfont=dict(size=10, color='black'),
            showlegend=False # Evita dezenas de legendas repetidas na direita
        ), row=count_row, col=count_cols)
        
        fig.update_xaxes(title_text="Distance", row=count_row, col=count_cols)
        fig.update_yaxes(title_text="Variogram", row=count_row, col=count_cols)
        
        count_cols += 1
        if count_cols > 4:
            count_cols = 1
            count_row += 1
            
    fig.update_layout(template="plotly_white", height=350*size_row)
    return fig

def plot_hscat(store, lagsize, lagmultiply, figsize=(800, 800)):
    """Plota o H-Scatterplot para visualização da correlação em diferentes lags."""
    distance = [i * lagsize for i in lagmultiply]
    fig = go.Figure()
    statistics_text = ""

    for j, i in enumerate(store):
        # CORREÇÃO: Exige > 1 para que a estatística de linregress não falhe
        if len(i[0]) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(i[0], i[1])
            statistics_text += f"rho ({np.round(r_value,2)}) / distance {distance[j]} <br>"
            x_val = np.linspace(0, np.max(i[0]), 100)
            y_val = slope * x_val + intercept
        else:
            x_val, y_val = [], []

        fig.add_trace(go.Scatter(
            x=i[0], y=i[1], mode='markers', name=f'Lag {distance[j]}',
            marker=dict(opacity=0.6)
        ))
        
        if len(x_val) > 0:
            fig.add_trace(go.Scatter(
                x=x_val, y=y_val, mode='lines', name=f'Reg: {distance[j]}',
                line=dict(width=3, dash='dash')
            ))

    fig.update_layout(
        title='H-Scatterplots',
        xaxis_title='Z(x)',
        yaxis_title='Z(x+h)',
        width=figsize[0], height=figsize[1],
        template="plotly_white",
        annotations=[dict(text=statistics_text, showarrow=False, x=0.02, y=0.98, xref='paper', yref='paper', align='left', bgcolor='white', bordercolor='black')]
    )
    return fig

def plot_locmap(x, y, variable, categorical=False, title='Location Map', x_axis='Easting (m)', y_axis='Northing (m)', pointsize=8, colorscale='Jet', colorbartitle='', figsize=(700, 600)):
    """Plota mapa de localização 2D."""
    variable = np.where(variable == -999.0, float('nan'), variable)
    traces = []
    
    if categorical:
        cats = np.unique(variable[~np.isnan(variable)])
        for cat in cats:
            mask = variable == cat
            traces.append(go.Scatter(
                x=x[mask], y=y[mask], mode='markers',
                marker=dict(size=pointsize), text=variable[mask], name=str(int(cat))
            ))
    else:
        traces.append(go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(size=pointsize, color=variable, colorscale=colorscale, showscale=True, colorbar=dict(title=colorbartitle)),
            text=variable, name='Data'
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title, xaxis=dict(title=x_axis, scaleanchor='y'), yaxis=dict(title=y_axis),
        width=figsize[0], height=figsize[1], template="plotly_white"
    )
    return fig

def plot_histogram(data, n_bins=20, wt=None, title='Histogram', x_axis='Value', y_axis='Frequency', cdf=False, figsize=(600, 500)):
    """Plota histograma e curva acumulada opcional."""
    dataf = np.where(data == -999.0, float('nan'), data)
    dataf = data[~np.isnan(data)]

    if wt is not None:
        mean, var = weighted_avg_and_var(dataf, wt)
        stat_type = "weighted"
    else:
        mean, var = dataf.mean(), dataf.var()
        stat_type = "unweighted"

    statistics = f"""
    n: {len(dataf)} <br>
    min: {round(dataf.min(),2)} <br>
    max: {round(dataf.max(),2)} <br>
    mean: {round(mean,2)} <br>
    stdev: {round(np.sqrt(var),2)} <br>
    cv: {round(np.sqrt(var)/mean, 2)} <br>
    {stat_type}
    """

    hist, bin_edges = np.histogram(dataf, bins=n_bins, weights=wt, density=True)
    hist = hist * np.diff(bin_edges)
    
    traces = [go.Bar(x=bin_edges, y=hist, name='PDF', marker_color='#1f77b4')]

    if cdf:
        traces.append(go.Scatter(x=bin_edges, y=np.cumsum(hist), name='CDF', mode='lines+markers', line=dict(color='red')))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title, xaxis_title=x_axis, yaxis_title=y_axis,
        width=figsize[0], height=figsize[1], barmode='overlay', template="plotly_white",
        annotations=[dict(text=statistics, showarrow=False, x=0.98, y=0.98, xref='paper', yref='paper', align='left', bgcolor='white', bordercolor='black')]
    )
    return fig

def plot_scatter2d(x, y, variable='kernel density', xy_line=True, best_fit_line=True, title='Scatter Plot', x_axis='X', y_axis='Y', pointsize=8, colorscale='Viridis', colorbartitle='', figsize=(800, 800)):
    """Gráfico de dispersão 2D com regressão linear."""
    x = np.where(x == -999.0, float('nan'), x)
    y = np.where(y == -999.0, float('nan'), y)
    x, y = isotopic_arrays([x, y])[0], isotopic_arrays([x, y])[1]

    statistics = f"n: {len(x)} <br>rho: {round(np.corrcoef([x,y])[1,0],2)}"
    traces = []

    if best_fit_line:
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        statistics += f"<br>slope: {round(slope,2)}<br>intercept: {round(intercept,2)}"
    
    if isinstance(variable, str):
        xy = np.vstack([x, y])
        variable = gaussian_kde(xy)(xy)
    else:
        variable = np.where(variable == -999.0, float('nan'), variable)
    
    traces.append(go.Scatter(
        x=x, y=y, mode='markers', name='Scatter',
        marker=dict(size=pointsize, color=variable, colorscale=colorscale, showscale=True, colorbar=dict(title=colorbartitle)),
        text=variable
    ))

    maxxy, minxy = max(max(x), max(y)), min(min(x), min(y))
    
    if xy_line:
        traces.append(go.Scatter(x=[minxy, maxxy], y=[minxy, maxxy], mode='lines', name='x=y line', line=dict(dash='dot', color='red')))

    if best_fit_line:
        vals = np.linspace(minxy, maxxy, 100)
        traces.append(go.Scatter(x=vals, y=slope*vals+intercept, mode='lines', name='Best Fit', line=dict(dash='dash', color='grey')))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title, xaxis_title=x_axis, yaxis_title=y_axis,
        width=figsize[0], height=figsize[1], template="plotly_white",
        annotations=[dict(text=statistics, showarrow=False, x=0.02, y=0.98, xref='paper', yref='paper', align='left', bgcolor='white', bordercolor='black')]
    )
    return fig

def plot_xval(real, estimate, error, pointsize=8, figsize=(1000, 400)):
    """Validação Cruzada (Cross-validation)."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=("True vs Estimates", "Error Histogram", "True vs Error"))

    real = np.where(real == -999.0, float('nan'), real)
    estimate = np.where(estimate == -999.0, float('nan'), estimate)
    slope, intercept, r_value, _, _ = stats.linregress(real, estimate)

    # 1. True x Estimates
    fig.add_trace(go.Scatter(x=real, y=estimate, mode='markers', marker=dict(size=pointsize, opacity=0.6), name='Estimates'), row=1, col=1)
    
    maxxy, minxy = max(max(real), max(estimate)), min(min(real), min(estimate))
    vals = np.linspace(minxy, maxxy, 100)
    fig.add_trace(go.Scatter(x=vals, y=slope*vals+intercept, mode='lines', line=dict(dash='dot', color='grey'), name='Fit'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[minxy, maxxy], y=[minxy, maxxy], mode='lines', line=dict(dash='dash', color='red'), name='x=y'), row=1, col=1)

    # 2. Histogram
    hist, bin_edges = np.histogram(error, bins=20, density=True)
    hist = hist * np.diff(bin_edges)
    fig.add_trace(go.Bar(x=bin_edges, y=hist, marker_color='#1f77b4', name='Error Dist'), row=1, col=2)

    # 3. True x Error
    fig.add_trace(go.Scatter(x=real, y=error, mode='markers', marker=dict(size=pointsize, opacity=0.6), name='Error'), row=1, col=3)

    stat_text = f"N: {len(error)}<br>Mean Err: {round(error.mean(),3)}<br>Std Err: {round(np.sqrt(error.var()),3)}<br>Rho: {round(r_value,3)}"
    fig.update_layout(
        title='Cross Validation Summary', width=figsize[0], height=figsize[1], template="plotly_white",
        annotations=[dict(text=stat_text, showarrow=False, x=0.35, y=0.95, xref='paper', yref='paper', align='left', bgcolor='white', bordercolor='black')]
    )
    return fig

def plot_swath_plots(x, y, z, point_var, grid_dic, grid_var, n_bins=10):
    """Gera gráficos de deriva (Swath plots) comparando dados originais com blocos krigados."""
    mask_pt = np.isfinite(point_var)
    if z is None: z = np.zeros(len(x))
    x, y, z = np.array(x)[mask_pt], np.array(y)[mask_pt], np.array(z)[mask_pt]
    point_var, grid_var = np.array(point_var)[mask_pt], np.array(grid_var)

    points_df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'var': point_var})
    
    # AQUI ESTÁ A MUDANÇA: Usando o utils nativamente
    coords = utils.add_coord(grid_dic)
    
    grid_df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
    grid_df['var'] = grid_var.flatten()

    fig = make_subplots(rows=3 if sum(z) != 0 else 2, cols=1, shared_xaxes=False)
    
    for i, axis in enumerate(['x', 'y', 'z']):
        if axis == 'z' and sum(z) == 0: continue
        
        bins = np.linspace(grid_df[axis].min(), grid_df[axis].max(), n_bins)
        pts_mean, grid_mean, n_pts = [], [], []

        for j in range(1, len(bins)):
            flt_pts = (points_df[axis] >= bins[j-1]) & (points_df[axis] <= bins[j])
            flt_grd = (grid_df[axis] >= bins[j-1]) & (grid_df[axis] <= bins[j])
            
            pts_mean.append(points_df.loc[flt_pts, 'var'].mean() if flt_pts.sum() > 0 else np.nan)
            grid_mean.append(grid_df.loc[flt_grd, 'var'].mean() if flt_grd.sum() > 0 else np.nan)
            n_pts.append(flt_pts.sum())

        row = i + 1
        fig.add_trace(go.Scatter(x=bins[1:], y=pts_mean, mode='lines+markers', name=f'Points ({axis})', line=dict(color='red')), row=row, col=1)
        fig.add_trace(go.Scatter(x=bins[1:], y=grid_mean, mode='lines+markers', name=f'Grid ({axis})', line=dict(color='blue')), row=row, col=1)
        
        fig.update_xaxes(title_text=f"Coordinate {axis.upper()}", row=row, col=1)
        fig.update_yaxes(title_text="Mean Value", row=row, col=1)

    fig.update_layout(title="Swath Plots (Drift Validation)", height=700, template="plotly_white")
    return fig
def plot_qqplot(data1, data2, title='QQ-Plot', x_axis='Var 1', y_axis='Var 2', figsize=(800, 800)):
    """Gera um gráfico de Quantil-Quantil para comparar duas variáveis."""
    data1, data2 = np.asarray(data1), np.asarray(data2)
    data1, data2 = data1[~np.isnan(data1)], data2[~np.isnan(data2)]
    
    # Calcula os decis/percentis (100 quebras)
    q = np.linspace(0, 1, 100)
    q1 = np.quantile(data1, q)
    q2 = np.quantile(data2, q)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q1, y=q2, mode='markers', name='QQ', marker=dict(color='#1f77b4')))
    
    min_val, max_val = min(min(q1), min(q2)), max(max(q1), max(q2))
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Referência (x=y)', line=dict(dash='dot', color='red')))
    
    fig.update_layout(title=title, xaxis_title=x_axis, yaxis_title=y_axis, width=figsize[0], height=figsize[1], template="plotly_white")
    return fig

def plot_varmap(df_varmap, title="Variogram Map (Varmap)", figsize=(10, 8)):
    """Gera o Varmap usando projeção polar do Matplotlib (Padrão Ouro Geoestatístico)"""
    r, theta, z = df_varmap['r'].values, df_varmap['theta'].values, df_varmap['valor'].values

    # Remove NaNs de direções não calculáveis
    mask = ~np.isnan(z)
    r, theta, z = r[mask], theta[mask], z[mask]

    # --- CORREÇÃO AQUI: Trava de segurança para interpolação ---
    if len(r) < 4:
        raise ValueError(f"Pontos insuficientes para o Varmap (encontrados apenas {len(r)}). Aumente o 'Lag Size' ou a 'Tol. Azim'.")

    # Cria uma grade regular para interpolar e desenhar o mapa de calor
    ri = np.linspace(0, r.max(), 50)
    ti = np.linspace(0, 2 * np.pi, 72)
    R, T = np.meshgrid(ri, ti)

    # Interpola em X, Y cartesianos para não haver "cortes" entre 0 e 360 graus
    x, y = r * np.sin(theta), r * np.cos(theta)
    xi, yi = R * np.sin(T), R * np.cos(T)
    zi = griddata((x, y), z, (xi, yi), method='linear')

    # Configura a Projeção Polar
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location("N") # Norte = 0 graus
    ax.set_theta_direction(-1)      # Gira no sentido horário

    cf = ax.contourf(T, R, zi, levels=40, cmap='jet')
    fig.colorbar(cf, ax=ax, pad=0.1)
    ax.set_title(title, va='bottom', pad=20)
    plt.close(fig) # Fecha para enviar apenas a figura ao Panel
    return fig
