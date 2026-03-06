import panel as pn
import numpy as np
import plotly.graph_objects as go
import os
import traceback
import logging
import warnings

# Nossas bibliotecas
from . import gslib
from . import plots
from . import variografia

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
    
    # A MÁGICA: Múltiplos Azimutes separados por vírgula!
    azm_input = pn.widgets.TextInput(name='Azimutes (ex: 0, 45, 90)', value='0, 90', width=180)
    
    atol = pn.widgets.FloatInput(name='Tol. Azim', value=22.5, start=0.0, end=90.0, width=100)
    bandwh = pn.widgets.FloatInput(name='Bandwidth', value=1000.0, start=0.0, width=100)
    omni_check = pn.widgets.Checkbox(name='Omnidirecional', value=False)
    flipped_check = pn.widgets.Checkbox(name='Flipped (1-Val)', value=False)
    
    # Botões de Ação
    run_var_btn = pn.widgets.Button(name='🚀 Calcular Variogramas', button_type='primary', width=200)
    run_varmap_btn = pn.widgets.Button(name='🎯 Gerar Varmap (Lento)', button_type='warning', width=200)
    run_hscat_btn = pn.widgets.Button(name='☁️ Gerar H-Scatter', button_type='primary', width=200)
    
    step_varmap = pn.widgets.IntInput(name='Passo Angular (°)', value=15, width=120)
    lags_hscat = pn.widgets.TextInput(name='Lags para H-Scatter (ex: 1, 2, 3)', value='1, 2, 3', width=220)

    # 2. Widgets Teóricos (Adicionado Anisotropia para Projetar as Curvas)
    enable_model = pn.widgets.Checkbox(name='Habilitar Modelo Teórico', value=False)
    mod_nugget = pn.widgets.FloatSlider(name='Efeito Pepita (C0)', start=0.0, end=1.0, value=0.0, step=0.01)
    mod_azm = pn.widgets.FloatSlider(name='Azimute Maior Continuidade', start=0.0, end=360.0, value=0.0, step=1.0)
    mod_s1_type = pn.widgets.Select(name='Estrutura 1', options=['Spherical', 'Exponential', 'Gaussian'], value='Spherical', width=120)
    mod_s1_cc = pn.widgets.FloatSlider(name='Contribuição (C1)', start=0.0, end=1.0, value=0.5, step=0.01)
    mod_s1_a_max = pn.widgets.FloatSlider(name='Alcance Maior (Dir Principal)', start=0.0, end=100.0, value=10.0, step=1.0)
    mod_s1_a_min = pn.widgets.FloatSlider(name='Alcance Menor (Ortogonal)', start=0.0, end=100.0, value=10.0, step=1.0)
    
    controles_teoricos = pn.Column(mod_nugget, pn.Row(mod_azm, mod_s1_a_max, mod_s1_a_min), pn.Row(mod_s1_type, mod_s1_cc), visible=False)

    status_var = pn.pane.Markdown("**Status:** Ajuste os parâmetros.", styles={'color': '#0056b3'})
    
    # Painéis (Note que o Varmap agora é um Matplotlib Pane!)
    pane_var = pn.pane.Plotly(sizing_mode="stretch_both", min_height=450)
    pane_varmap = pn.pane.Matplotlib(sizing_mode="stretch_both", min_height=700, tight=True)
    pane_hscat = pn.pane.Plotly(sizing_mode="stretch_both", min_height=600)

    # --- Callbacks ---
    @pn.depends(mod_nugget.param.value_throttled, mod_azm.param.value_throttled, mod_s1_type, mod_s1_cc.param.value_throttled, mod_s1_a_max.param.value_throttled, mod_s1_a_min.param.value_throttled, enable_model, watch=True)
    def atualizar_plot_teorico(*args):
        df_exp_list = geo_state.get('df_exp_list')
        azm_list = geo_state.get('azm_list')
        if not df_exp_list: return
        
        controles_teoricos.visible = enable_model.value
        
        # Redesenha todos os variogramas experimentais primeiro
        fig = plots.plot_experimental_variogram(df_exp_list, azm_list, [0.0]*len(azm_list))
        
        # Injeta as curvas teóricas projetadas em cada subplot
        if enable_model.value:
            count_row, count_cols = 1, 1
            for i, (df_exp, azm_plot) in enumerate(zip(df_exp_list, azm_list)):
                df_teorico = variografia.calcular_modelo_teorico(
                    experimental_dataframe=df_exp, azimuth=azm_plot, dip=0.0, 
                    rotation_reference=[mod_azm.value, 0, 0], model_func=[mod_s1_type.value], 
                    ranges=[[mod_s1_a_max.value, mod_s1_a_min.value, mod_s1_a_max.value]], # Y(Max), X(Min), Z
                    contribution=[mod_s1_cc.value], nugget=mod_nugget.value, inverted=False
                )
                fig.add_trace(go.Scatter(x=df_teorico['distances'], y=df_teorico['model'], mode='lines', name=f'Teórico {azm_plot}°', line=dict(color='red', width=3)), row=count_row, col=count_cols)
                
                count_cols += 1
                if count_cols > 4: count_cols = 1; count_row += 1
                
        pane_var.object = fig

    def btn_calc_var(event):
        if geo_state['df'] is None: return
        status_var.object = "⏳ Calculando Variogramas..."
        try:
            df, x, y, v = geo_state['df'], geo_state['x_col'], geo_state['y_col'], geo_state['var_col']
            
            # Interpreta a lista de azimutes do usuário
            azm_list = [float(a.strip()) for a in azm_input.value.split(',')]
            ndir = len(azm_list)
            
            # Executa N direções de uma vez!
            res = variografia.experimental(
                df.copy(), x, y, v, v, [type_var.value]*ndir, [nlag.value]*ndir, [xlag.value]*ndir, 
                [xlag.value/2.0]*ndir, [atol.value]*ndir, [90.0]*ndir, [bandwh.value]*ndir, 
                [1000.0]*ndir, azm_list, [0.0]*ndir, [omni_check.value]*ndir, [flipped_check.value]*ndir, Z=None
            )
            geo_state['df_exp_list'] = res['Values']
            geo_state['azm_list'] = azm_list
            
            # Atualiza os limites dos sliders baseado no maior variograma
            max_g = max([d['Spatial continuity'].max() for d in geo_state['df_exp_list'] if not d.empty])
            max_d = max([d['Average distance'].max() for d in geo_state['df_exp_list'] if not d.empty])
            mod_nugget.end, mod_s1_cc.end = float(max_g*1.5), float(max_g*1.5)
            mod_s1_a_max.end, mod_s1_a_min.end = float(max_d*1.5), float(max_d*1.5)
            
            atualizar_plot_teorico()
            status_var.object = "✅ Múltiplos Variogramas calculados!"
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
            # Pega o primeiro Azimute da lista para o H-scatter
            azm_ref = float(azm_input.value.split(',')[0].strip())
            df, x, y, v = geo_state['df'], geo_state['x_col'], geo_state['y_col'], geo_state['var_col']
            h_data = variografia.calcular_hscatter_data(df.copy(), x, y, v, v, nlag.value, xlag.value, xlag.value/2.0, atol.value, 90.0, bandwh.value, 1000.0, azm_ref, 0.0, omni_check.value, Z=None)
            
            # Filtra apenas os lags que o usuário pediu!
            try: lags_to_plot = [int(l.strip()) for l in lags_hscat.value.split(',')]
            except: lags_to_plot = [1, 2, 3] 
            
            store, lagmult = [], []
            for i in lags_to_plot:
                lk = f'Lag_{i}'
                if lk in h_data and h_data[lk]['n_pares'] > 0:
                    store.append([h_data[lk]['head'], h_data[lk]['tail']])
                    lagmult.append(i)
                    
            pane_hscat.object = plots.plot_hscat(store, xlag.value, lagmult)
            status_var.object = f"✅ H-Scatter gerado para o Azimute {azm_ref}°!"
        except Exception as e: status_var.object = f"❌ Erro:\n```\n{traceback.format_exc()}\n```"
    run_hscat_btn.on_click(btn_calc_hscat)

    # Sub-abas da Variografia
    tabs_var = pn.Tabs(
        ('📈 Múltiplos Variogramas', pn.Column(run_var_btn, enable_model, controles_teoricos, pane_var)),
        ('🎯 Varmap', pn.Column(pn.Row(step_varmap, run_varmap_btn), pane_varmap)),
        ('☁️ H-Scatter', pn.Column(pn.Row(lags_hscat, run_hscat_btn), pane_hscat)),
        dynamic=True
    )

    return pn.Column(
        "## 📈 2. Continuidade Espacial",
        pn.Row(type_var, nlag, xlag, azm_input, atol, bandwh),
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
