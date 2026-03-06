import panel as pn
import numpy as np
import plotly.graph_objects as go
import os
import traceback
import logging
import warnings

# Nossas bibliotecas
import gslib
import plots
import variografia

warnings.filterwarnings('ignore')
logging.getLogger('bokeh').setLevel(logging.ERROR)
logging.getLogger('param').setLevel(logging.ERROR)

pn.config.theme = 'default'
pn.extension('plotly', sizing_mode="stretch_width")

# ==============================================================================
# ESTADO GLOBAL
# ==============================================================================
geo_state = {'df': None, 'x_col': None, 'y_col': None, 'var_col': None, 'var_sec': None, 'df_exp': None}

def listar_arquivos_dat():
    return [f for f in os.listdir('.') if os.path.isfile(f) and not f.startswith('.')]

# ==============================================================================
# MÓDULO 1: INTERFACE DA ANÁLISE EXPLORATÓRIA (EDA)
# ==============================================================================
def renderizar_eda():
    refresh_btn = pn.widgets.Button(name='🔄 Atualizar Lista', button_type='default', width=150)
    file_data = pn.widgets.Select(name="Selecione o Dataset (GSLIB)", options=[''] + listar_arquivos_dat())
    load_btn = pn.widgets.Button(name='Ler Dataset', button_type='primary', width=150)
    
    x_col = pn.widgets.Select(name='Coordenada X', options=[])
    y_col = pn.widgets.Select(name='Coordenada Y', options=[])
    var_col = pn.widgets.Select(name='Variável Principal', options=[])
    var_sec = pn.widgets.Select(name='Variável Secundária (Para Scatter/QQ)', options=[])
    eda_btn = pn.widgets.Button(name='Gerar EDA', button_type='success', width=150)
    status_text = pn.pane.Markdown("**Status:** Aguardando arquivo...", styles={'color': '#0056b3'})
    
    # Painéis de Plotagem
    pane_mapa = pn.pane.Plotly(sizing_mode="stretch_both")
    pane_hist = pn.pane.Plotly(sizing_mode="stretch_both")
    pane_scatter = pn.pane.Plotly(sizing_mode="stretch_both")
    pane_qq = pn.pane.Plotly(sizing_mode="stretch_both")

    # Abas da EDA
    tabs_eda = pn.Tabs(
        ('📍 Mapa & Histograma', pn.Row(pane_mapa, pane_hist, min_height=450)),
        ('📉 Dispersão & QQ-Plot', pn.Row(pane_scatter, pane_qq, min_height=450)),
        dynamic=True
    )

    def atualizar_lista(event):
        file_data.options = [''] + listar_arquivos_dat()
        status_text.object = "**Status:** 🔄 Lista atualizada!"
    refresh_btn.on_click(atualizar_lista)

    def carregar_dados(event):
        if not file_data.value: return
        status_text.object = f"**Status:** ⏳ Lendo `{file_data.value}`..."
        try:
            df = gslib.read_gslib(file_data.value)
            cols = df.columns.tolist()
            x_col.options = y_col.options = var_col.options = var_sec.options = cols
            if 'X' in cols: x_col.value = 'X'
            if 'Y' in cols: y_col.value = 'Y'
            if len(cols) > 2: var_col.value = cols[2]
            if len(cols) > 3: var_sec.value = cols[3]
            
            geo_state['df'] = df
            status_text.object = "**Status:** ✅ Dados lidos! Configure as variáveis e clique em Gerar EDA."
        except Exception as e:
            status_text.object = f"**Status:** ❌ Erro: {e}"
    load_btn.on_click(carregar_dados)

    def gerar_graficos(event):
        if geo_state['df'] is None: return
        status_text.object = "**Status:** ⏳ Desenhando gráficos..."
        try:
            geo_state['x_col'], geo_state['y_col'] = x_col.value, y_col.value
            geo_state['var_col'], geo_state['var_sec'] = var_col.value, var_sec.value
            df = geo_state['df']
            
            # Aba 1
            pane_mapa.object = plots.plot_locmap(df[x_col.value].values, df[y_col.value].values, df[var_col.value].values, title=f"Mapa: {var_col.value}")
            pane_hist.object = plots.plot_histogram(df[var_col.value].values, title=f"Histograma: {var_col.value}")
            
            # Aba 2
            pane_scatter.object = plots.plot_scatter2d(df[var_col.value].values, df[var_sec.value].values, title="Scatter Plot", x_axis=var_col.value, y_axis=var_sec.value)
            pane_qq.object = plots.plot_qqplot(df[var_col.value].values, df[var_sec.value].values, title="QQ-Plot", x_axis=var_col.value, y_axis=var_sec.value)
            
            status_text.object = "**Status:** ✅ EDA pronta!"
        except Exception as e:
            status_text.object = f"**Status:** ❌ Erro na EDA:\n```\n{traceback.format_exc()}\n```"
    eda_btn.on_click(gerar_graficos)

    return pn.Column(
        "## 📊 1. Análise Exploratória (EDA)",
        pn.Row(refresh_btn, file_data, load_btn, align='end'),
        pn.Row(x_col, y_col, var_col, var_sec, eda_btn, align='end'),
        status_text, pn.layout.Divider(), tabs_eda,
        styles={'background': '#f9f9f9', 'padding': '20px', 'border-radius': '10px'}
    )

# ==============================================================================
# MÓDULO 2: INTERFACE DA VARIOGRAFIA
# ==============================================================================
def renderizar_variografia():
    # 1. Widgets Experimentais (Painel Lateral/Superior)
    type_var = pn.widgets.Select(name='Função', options=['Variogram', 'Covariogram', 'Correlogram', 'Non_Ergodic_Correlogram', 'PairWise', 'Relative_Variogram'], value='Variogram', width=200)
    nlag = pn.widgets.IntInput(name='N Lags', value=15, start=1, width=100)
    xlag = pn.widgets.FloatInput(name='Lag Size', value=10.0, start=0.1, width=100)
    azm = pn.widgets.FloatInput(name='Azimute', value=0.0, start=0.0, end=360.0, width=100)
    atol = pn.widgets.FloatInput(name='Tol. Azim', value=22.5, start=0.0, end=90.0, width=100)
    bandwh = pn.widgets.FloatInput(name='Bandwidth', value=1000.0, start=0.0, width=100)
    omni_check = pn.widgets.Checkbox(name='Omnidirecional', value=False)
    flipped_check = pn.widgets.Checkbox(name='Flipped (1-Val)', value=False)
    
    # Botões de Ação para as Abas
    run_var_btn = pn.widgets.Button(name='🚀 Calcular Variograma', button_type='primary', width=200)
    run_varmap_btn = pn.widgets.Button(name='🎯 Gerar Varmap (Lento)', button_type='warning', width=200)
    run_hscat_btn = pn.widgets.Button(name='☁️ Gerar H-Scatter', button_type='primary', width=200)
    
    step_varmap = pn.widgets.IntInput(name='Passo Angular (°)', value=15, width=120)

    # 2. Widgets Teóricos
    enable_model = pn.widgets.Checkbox(name='Habilitar Modelo Teórico', value=False)
    mod_nugget = pn.widgets.FloatSlider(name='Efeito Pepita (C0)', start=0.0, end=1.0, value=0.0, step=0.01)
    mod_s1_type = pn.widgets.Select(name='Estrutura 1', options=['Spherical', 'Exponential', 'Gaussian'], value='Spherical', width=120)
    mod_s1_cc = pn.widgets.FloatSlider(name='Contribuição (C1)', start=0.0, end=1.0, value=0.5, step=0.01)
    mod_s1_a = pn.widgets.FloatSlider(name='Alcance (a1)', start=0.0, end=100.0, value=10.0, step=1.0)
    controles_teoricos = pn.Column(mod_nugget, pn.Row(mod_s1_type, mod_s1_cc, mod_s1_a), visible=False)

    status_var = pn.pane.Markdown("**Status:** Ajuste os parâmetros e escolha a aba desejada.", styles={'color': '#0056b3'})
    
    # Painéis
    pane_var = pn.pane.Plotly(sizing_mode="stretch_both", min_height=450)
    pane_varmap = pn.pane.Plotly(sizing_mode="stretch_both", min_height=600)
    pane_hscat = pn.pane.Plotly(sizing_mode="stretch_both", min_height=600)

    # --- Callbacks ---
    @pn.depends(mod_nugget.param.value_throttled, mod_s1_type, mod_s1_cc.param.value_throttled, mod_s1_a.param.value_throttled, enable_model, watch=True)
    def atualizar_plot_teorico(*args):
        df_exp = geo_state.get('df_exp')
        if df_exp is None or df_exp.empty: return
        controles_teoricos.visible = enable_model.value
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_exp['Average distance'], y=df_exp['Spatial continuity'], mode='markers', name='Experimental', marker=dict(size=10, color=df_exp['Number of pairs'], colorscale='Viridis', showscale=True), text=df_exp['Number of pairs']))
        
        if enable_model.value:
            df_teorico = variografia.calcular_modelo_teorico(
                experimental_dataframe=df_exp, azimuth=azm.value, dip=0.0, rotation_reference=[azm.value, 0, 0], 
                model_func=[mod_s1_type.value], ranges=[[mod_s1_a.value, mod_s1_a.value, mod_s1_a.value]], 
                contribution=[mod_s1_cc.value], nugget=mod_nugget.value, inverted=False
            )
            fig.add_trace(go.Scatter(x=df_teorico['distances'], y=df_teorico['model'], mode='lines', name='Teórico', line=dict(color='red', width=3)))
        
        fig.update_layout(title=f"Variograma ({type_var.value})", xaxis_title="Distância (h)", yaxis_title="\u03B3 (h)", template="plotly_white")
        pane_var.object = fig

    def btn_calc_var(event):
        if geo_state['df'] is None: return
        status_var.object = "⏳ Calculando Variograma..."
        try:
            df, x, y, v = geo_state['df'], geo_state['x_col'], geo_state['y_col'], geo_state['var_col']
            res = variografia.experimental(df.copy(), x, y, v, v, [type_var.value], [nlag.value], [xlag.value], [xlag.value/2.0], [atol.value], [90.0], [bandwh.value], [1000.0], [azm.value], [0.0], [omni_check.value], [flipped_check.value], Z=None)
            geo_state['df_exp'] = res['Values'][0]
            if geo_state['df_exp'].empty: status_var.object = "⚠️ Sem pares."; return
            
            max_g = geo_state['df_exp']['Spatial continuity'].max()
            max_d = geo_state['df_exp']['Average distance'].max()
            mod_nugget.end, mod_s1_cc.end, mod_s1_a.end = float(max_g*1.5), float(max_g*1.5), float(max_d*1.5)
            atualizar_plot_teorico()
            status_var.object = "✅ Variograma concluído!"
        except Exception as e: status_var.object = f"❌ Erro:\n```\n{traceback.format_exc()}\n```"
    run_var_btn.on_click(btn_calc_var)

    def btn_calc_varmap(event):
        if geo_state['df'] is None: return
        status_var.object = "⏳ Varrendo 360º para o Varmap (Isso pode levar alguns segundos)..."
        try:
            df, x, y, v = geo_state['df'], geo_state['x_col'], geo_state['y_col'], geo_state['var_col']
            df_vm = variografia.calcular_varmap_data(df.copy(), x, y, v, v, type_var.value, nlag.value, xlag.value, xlag.value/2.0, atol.value, 90.0, bandwh.value, 1000.0, dip=0.0, step_angulo=step_varmap.value, Z=None, flipped=flipped_check.value)
            pane_varmap.object = plots.plot_varmap(df_vm, title=f"Varmap ({type_var.value})")
            status_var.object = "✅ Varmap gerado!"
        except Exception as e: status_var.object = f"❌ Erro:\n```\n{traceback.format_exc()}\n```"
    run_varmap_btn.on_click(btn_calc_varmap)

    def btn_calc_hscat(event):
        if geo_state['df'] is None: return
        status_var.object = "⏳ Extraindo pares para o H-Scatterplot..."
        try:
            df, x, y, v = geo_state['df'], geo_state['x_col'], geo_state['y_col'], geo_state['var_col']
            h_data = variografia.calcular_hscatter_data(df.copy(), x, y, v, v, nlag.value, xlag.value, xlag.value/2.0, atol.value, 90.0, bandwh.value, 1000.0, azm.value, 0.0, omni_check.value, Z=None)
            
            store, lagmult = [], []
            for i in range(1, nlag.value + 1):
                lk = f'Lag_{i}'
                if lk in h_data and h_data[lk]['n_pares'] > 0:
                    store.append([h_data[lk]['head'], h_data[lk]['tail']])
                    lagmult.append(i)
                    
            pane_hscat.object = plots.plot_hscat(store, xlag.value, lagmult)
            status_var.object = "✅ H-Scatter gerado!"
        except Exception as e: status_var.object = f"❌ Erro:\n```\n{traceback.format_exc()}\n```"
    run_hscat_btn.on_click(btn_calc_hscat)

    # Sub-abas da Variografia
    tabs_var = pn.Tabs(
        ('📈 Variograma', pn.Column(run_var_btn, enable_model, controles_teoricos, pane_var)),
        ('🎯 Varmap', pn.Column(pn.Row(step_varmap, run_varmap_btn), pane_varmap)),
        ('☁️ H-Scatter', pn.Column(run_hscat_btn, pane_hscat)),
        dynamic=True
    )

    return pn.Column(
        "## 📈 2. Continuidade Espacial",
        pn.Row(type_var, nlag, xlag, azm, atol, bandwh),
        pn.Row(omni_check, flipped_check),
        status_var, pn.layout.Divider(), tabs_var,
        styles={'background': '#f9f9f9', 'padding': '20px', 'border-radius': '10px'}
    )

# ==============================================================================
# ORQUESTRADOR: APP PRINCIPAL
# ==============================================================================
def App():
    return pn.Tabs(
        ('1. Exploratória (EDA)', renderizar_eda()),
        ('2. Variografia', renderizar_variografia()),
        dynamic=True
    )
