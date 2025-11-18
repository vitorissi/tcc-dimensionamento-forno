import sys
import os

# Adiciona o diretório atual (onde app.py está) ao path do Python
# Isso garante que ele encontre a pasta 'tcc'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataclasses import asdict

# Importa os módulos principais do pacote TCC
try:
    from tcc.data_types import DimensoesForno, Temperaturas, Material
    from tcc.materials import MATERIAIS_DISPONIVEIS
    from tcc.calc import (
        propriedades_do_ar,
        coeficiente_conveccao_natural,
        taxa_perda_sup_plana,  
        fator_de_forma,
        troca_radiacao_superficies, 
        resistencia_total,
        calc_espessura,
        grashof,
        numero_de_raylength,
        nusselt_lateral,
        coeficiente_conveccao_natural_lateral,
        calc_raio_intermediario,
        calc_volume_parede_resistencia,
        analise_lateral,
        calcular_dimensoes_externas,
        calcular_correntes_eletricas,
        calc_volume_parede_plana,
        calc_volume_parede_cilindrica,
        calc_custo_total_material,
    )
except ImportError as e:
    st.error(
        f"Erro Crítico de Importação: {e}. "
        "Verifique se 'app.py' está no mesmo nível da pasta 'tcc' "
        "e se a pasta 'tcc' contém o arquivo '__init__.py'."
    )
    st.stop()


# --- Funções de Lógica e Plotagem ---

@st.cache_data # <-- Adiciona cache para acelerar re-execuções
def executar_analise_completa(
    material_tampa: Material,
    material_base: Material,
    material_lateral: Material,
    material_resistencia: Material,
    dimensoes: DimensoesForno,
    temperaturas: Temperaturas,
    espessura_parede_resistencia_mm: float,
) -> dict | None:
    """
    Executa a simulação térmica completa para um conjunto de materiais.
    Retorna um dicionário com os resultados, ou None se falhar.
    """
    try:
        nomes_materiais = set(
            [material_tampa.nome, material_base.nome, material_lateral.nome]
        )
        if len(nomes_materiais) == 1:
            nome_analise = material_tampa.nome
        else:
            nome_analise = f"Lateral({material_lateral.nome})/Tampa({material_tampa.nome})/Base({material_base.nome})"

        # --- Início da lógica de cálculo ---
        propriedades_ar = propriedades_do_ar(temperaturas)
        h_ext_base, h_ext_tampa = coeficiente_conveccao_natural(
            dimensoes, temperaturas, propriedades_ar
        )

        espessura_parede_resistencia = espessura_parede_resistencia_mm / 1000.0

        # Analise tampa
        taxa_radiacao_int_tampa = taxa_perda_sup_plana( 
            dimensoes.area_tampa,
            h_ext_tampa,
            material_tampa.emissividade,
            temperaturas,
            propriedades_ar,
        )

        # Analise base
        taxa_radiacao_int_base = taxa_perda_sup_plana( 
            dimensoes.area_base,
            h_ext_base,
            material_base.emissividade,
            temperaturas,
            propriedades_ar,
        )

        f_bl, f_bt = fator_de_forma(dimensoes)

        resultado_radiacao = troca_radiacao_superficies( 
            temperaturas,
            material_base.emissividade,
            material_tampa.emissividade,
            material_lateral.emissividade,
            dimensoes,
            f_bl,
            f_bt,
            taxa_radiacao_int_base,
            taxa_radiacao_int_tampa,
        )

        if resultado_radiacao is None:
            st.error(f"Análise de radiação falhou para {nome_analise}.")
            return None

        t_base_interna, t_tampa_interna, q_radiacao_lateral_interna = resultado_radiacao

        # 6. Analise espessura Tampa e Base
        taxa_conducao_int_tampa = taxa_radiacao_int_tampa
        espessura_tampa = calc_espessura(
            material_tampa,
            t_tampa_interna,
            temperaturas.parede_externa,
            taxa_conducao_int_tampa,
            dimensoes.area_tampa,
        )

        taxa_conducao_int_base = taxa_radiacao_int_base
        espessura_base = calc_espessura(
            material_base,
            t_base_interna,
            temperaturas.parede_externa,
            taxa_conducao_int_base,
            dimensoes.area_base,
        )

        # 7. Analise Conveção natural exterior Lateral
        gr_l = grashof(
            temperaturas.parede_externa,
            temperaturas.ambiente,
            dimensoes.altura_interna,
            propriedades_ar.beta,
            propriedades_ar.nu,
        )
        ra_lateral = numero_de_raylength(gr_l, propriedades_ar.Pr)
        nusselt_lateral_value = nusselt_lateral(ra_lateral, propriedades_ar.Pr)
        h_ext_lateral = coeficiente_conveccao_natural_lateral(
            nusselt_lateral_value, propriedades_ar.k_ar, dimensoes.altura_interna
        )

        # 8. Analise espessura parede com resistencia
        raio_intermediario = calc_raio_intermediario(
            dimensoes.raio_interno, espessura_parede_resistencia
        )
        volume_parede_resistencia = calc_volume_parede_resistencia(
            raio_intermediario, dimensoes.raio_interno, dimensoes.altura_interna
        )

        # 9. Analise Lateral

        resultados_laterais = analise_lateral(
            espessura_parede_resistencia=espessura_parede_resistencia,
            temperaturas=temperaturas,
            dimensoes_forno=dimensoes,
            material_lateral=material_lateral,
            material_resistencia=material_resistencia,
            q_radiacao_lateral_interna=q_radiacao_lateral_interna,
            h_ext_lateral=h_ext_lateral,
            volume_parede_resistencia=volume_parede_resistencia,
            raio_intermediario=raio_intermediario,
            erro_adm=1e-4,
        )

        if resultados_laterais is None:
            st.error(f"Análise lateral falhou para {nome_analise}.")
            return None

        # --- Cálculo de Custo ---
        vol_tampa = calc_volume_parede_plana(dimensoes.area_tampa, espessura_tampa)
        custo_tampa = calc_custo_total_material(
            vol_tampa, material_tampa.custo_por_m3
        )

        vol_base = calc_volume_parede_plana(dimensoes.area_base, espessura_base)
        custo_base = calc_custo_total_material(vol_base, material_base.custo_por_m3)

        vol_lateral = calc_volume_parede_cilindrica(
            raio_interno=raio_intermediario,
            raio_externo=resultados_laterais.raio_externo_final,
            altura=dimensoes.altura_interna,
        )
        custo_lateral = calc_custo_total_material(
            vol_lateral, material_lateral.custo_por_m3
        )

        custo_total = custo_tampa + custo_base + custo_lateral

        # --- Coleta de Resultados ---
        return {
            "Material": nome_analise,
            "e_lateral_mm": resultados_laterais.espessura_isolamento_lateral * 1000,
            "e_base_mm": espessura_base * 1000,
            "e_tampa_mm": espessura_tampa * 1000,
            "Potencia_Total": resultados_laterais.potencia_total_necessaria,
            "Custo_Tampa": custo_tampa,
            "Custo_Base": custo_base,
            "Custo_Lateral": custo_lateral,
            "Custo_Total": custo_total,
            "T_intermediaria_C": resultados_laterais.temperatura_intermediaria - 273.15,
            "Perda_Lateral_W": resultados_laterais.perda_termica_lateral,
            "Mat_Lateral": material_lateral.nome,
            "Mat_Tampa": material_tampa.nome,
            "Mat_Base": material_base.nome,
        }

    except Exception as e:
        st.error(f"Um erro inesperado ocorreu durante a análise: {e}")
        import traceback
        st.exception(traceback.format_exc())
        return None

@st.cache_data # Cache para acelerar a repetição da análise
def calcular_potencia_sensibilidade(
    T_amb_C,  # A nova temperatura ambiente em °C para este ponto da simulação
    dimensoes,
    T_interna_K, # Temperatura interna (fixa)
    T_externa_K, # Temperatura da parede externa (fixa)
    materiais, # Tupla de (tampa, base, lateral, resistencia)
    espessuras_fixas # Tupla de (tampa_m, base_m, lateral_m, resistencia_m)
):
    """
    Recalcula a potência necessária para uma ÚNICA temperatura ambiente,
    com base em dimensões de forno JÁ CALCULADAS (fixas).
    """
    try:
        # 1. Criar novo objeto Temperaturas com a T_amb da iteração
        temp_iteracao = Temperaturas(
            ambiente=T_amb_C + 273.15,
            lateral_interna=T_interna_K, # Fixo
            parede_externa=T_externa_K   # Fixo
        )
        
        mat_tampa, mat_base, mat_lateral, mat_resistencia = materiais
        e_tampa, e_base, e_lateral, e_resistencia = espessuras_fixas

        # 2. Recalcular propriedades do ar e convecção (dependem da T_amb)
        prop_ar = propriedades_do_ar(temp_iteracao)
        h_ext_base, h_ext_tampa = coeficiente_conveccao_natural(dimensoes, temp_iteracao, prop_ar)

        # 3. Recalcular perdas de base e tampa (dependem da T_amb)
        q_rad_int_tampa = taxa_perda_sup_plana(dimensoes.area_tampa, h_ext_tampa, mat_tampa.emissividade, temp_iteracao, prop_ar)
        q_rad_int_base = taxa_perda_sup_plana(dimensoes.area_base, h_ext_base, mat_base.emissividade, temp_iteracao, prop_ar)

        # 4. Recalcular troca de radiação interna (depende das perdas externas)
        f_bl, f_bt = fator_de_forma(dimensoes)
        resultado_rad = troca_radiacao_superficies(
            temp_iteracao,
            mat_base.emissividade,
            mat_tampa.emissividade,
            mat_lateral.emissividade,
            dimensoes,
            f_bl,
            f_bt,
            q_rad_int_base,
            q_rad_int_tampa
        )
        if not resultado_rad:
            return None # Falha
        
        _, _, q_rad_lat_interna = resultado_rad

        # 5. Recalcular convecção lateral (depende da T_amb)
        gr_l = grashof(temp_iteracao.parede_externa, temp_iteracao.ambiente, dimensoes.altura_interna, prop_ar.beta, prop_ar.nu)
        ra_lateral = numero_de_raylength(gr_l, prop_ar.Pr)
        nusselt_lat = nusselt_lateral(ra_lateral, prop_ar.Pr)
        h_ext_lateral = coeficiente_conveccao_natural_lateral(nusselt_lat, prop_ar.k_ar, dimensoes.altura_interna)

        # 6. Recalcular perda lateral (com geometria FIXA)
        # Reconstruir geometria externa com base nas espessuras fixas
        raio_intermediario = calc_raio_intermediario(dimensoes.raio_interno, e_resistencia)
        raio_externo = raio_intermediario + e_lateral
        area_lateral_externa = 2 * np.pi * raio_externo * dimensoes.altura_interna

        # Usar taxa_perda_sup_plana para calcular a perda lateral total (convecção + radiação)
        q_cond_l = taxa_perda_sup_plana(area_lateral_externa, h_ext_lateral, mat_lateral.emissividade, temp_iteracao, prop_ar)

        # 7. Calcular Potência Final
        potencia_final = abs(q_rad_lat_interna) + q_cond_l
        return potencia_final
    
    except Exception as e:
        st.error(f"Erro na análise de sensibilidade para T_amb={T_amb_C}°C: {e}")
        return None

def plotar_grafico_espessuras(df: pd.DataFrame):
    """Gera o gráfico de barras comparativas e retorna a figura."""
    fig, ax = plt.subplots(figsize=(12, 6))

    materiais_nomes = df["Material"].tolist()
    n_materiais = len(materiais_nomes)
    
    espessuras_lateral = df["e_lateral_mm"]
    espessuras_base = df["e_base_mm"]
    espessuras_tampa = df["e_tampa_mm"]

    componentes = ["Lateral", "Base", "Tampa"]
    x = np.arange(len(componentes))
    
    # --- Lógica de Largura Dinâmica ---
    largura_total_grupo = 0.8  # Largura total para todas as barras de um grupo (ex: "Lateral")
    largura_barra_individual = largura_total_grupo / n_materiais
    # --- Fim da Lógica ---

    for i, material in enumerate(materiais_nomes):
        # Calcula o deslocamento (offset) de cada barra a partir do centro (x)
        offset = (i - n_materiais / 2) * largura_barra_individual + largura_barra_individual / 2
        pos = x + offset
        
        ax.bar(
            pos,
            [espessuras_lateral.iloc[i], espessuras_base.iloc[i], espessuras_tampa.iloc[i]],
            largura_barra_individual, # <-- Usa a largura dinâmica
            label=material,
        )

    ax.set_xticks(x) # <-- Centraliza o "tick" no meio do grupo
    ax.set_xticklabels(componentes)
    ax.set_ylabel("Espessura (mm)")
    ax.set_xlabel("Componente")
    ax.set_title("Comparativo de Espessura dos Isolantes")
    ax.legend(title="Material", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()
    return fig


def plotar_grafico_custo_potencia(df: pd.DataFrame):
    """Gera o gráfico de dispersão e retorna a figura."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if len(df) == 0:
        ax.text(0.5, 0.5, "Sem dados para exibir", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return fig
        
    try:
        colors = plt.cm.get_cmap("tab10", len(df))
    except ValueError: # Caso len(df) seja 0
        colors = plt.cm.get_cmap("tab10", 1)
        
    markers = ["o", "s", "^", "D", "P", "*", "X", "v", "<", ">"]

    for i, mat in df.iterrows():
        ax.scatter(
            mat["Custo_Total"],
            mat["Potencia_Total"],
            color=colors(i % 10), # Modulo 10 para evitar erro se mais de 10 materiais
            marker=markers[i % len(markers)],
            label=mat["Material"],
            s=100,
        )

    ax.set_xlabel("Custo Total (R$)")
    ax.set_ylabel("Potência Total Requerida (W)")
    ax.set_title("Potência Requerida vs. Custo Total do Isolamento")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)

    return fig

def plotar_grafico_sensibilidade_temp(temp_list_c, potencia_list_w):
    """Gera o gráfico de linha Potência x Temp. Ambiente."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(temp_list_c, potencia_list_w, 'r-', linewidth=2, marker='o')
    
    ax.set_xlabel('Temperatura Ambiente (°C)')
    ax.set_ylabel('Potência Necessária (W)')
    ax.set_title('Análise de Sensibilidade: Potência vs. Temperatura Ambiente')
    ax.grid(True)
    fig.tight_layout()
    return fig

def get_material_from_list(nome: str, lista_materiais: list[dict]) -> Material | None:
    """Encontra um material na lista de dicionários e o retorna como objeto Material."""
    for mat_dict in lista_materiais:
        if mat_dict["nome"] == nome:
            try:
                return Material(**mat_dict)
            except TypeError:
                st.error(f"Erro ao converter o material '{nome}'. "
                         f"Verifique se o banco de materiais tem colunas extras ou faltando: {mat_dict}")
                return None
    return None


# --- Configuração da Página ---

st.set_page_config(
    page_title="Análise de Forno",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔥 Ferramenta de Análise Térmica de Forno")

# --- Inicialização do Session State ---
if "materiais_db" not in st.session_state:
    st.session_state.materiais_db = [asdict(mat) for mat in MATERIAIS_DISPONIVEIS]


# --- Barra Lateral de Entradas (Comum a ambas as análises) ---

st.sidebar.header("Parâmetros de Entrada (Global)")
st.sidebar.markdown("Defina as condições de operação e geometria do forno.")

with st.sidebar.expander("Geometria do Forno", expanded=True):
    altura_interna_mm = st.number_input(
        "Altura Interna (mm)", min_value=1.0, value=450.0, step=10.0
    )
    diametro_interno_mm = st.number_input(
        "Diâmetro Interno (mm)", min_value=1.0, value=400.0, step=10.0
    )

with st.sidebar.expander("Temperaturas de Operação", expanded=True):
    temp_ambiente_c = st.number_input(
        "Temperatura Ambiente (°C)", min_value=-20.0, value=25.0, step=1.0
    )
    temp_interna_c = st.number_input(
        "Temperatura Interna Lateral (°C)", min_value=100.0, value=1250.0, step=10.0
    )
    temp_externa_c = st.number_input(
        "Temperatura Externa Alvo (°C)", min_value=30.0, value=75.0, step=1.0
    )

with st.sidebar.expander("Parede de Resistência", expanded=True):
    espessura_resistencia_mm = st.number_input(
        "Espessura Parede Resistência (mm)", min_value=1.0, value=63.0, step=1.0
    )
    
    material_nomes = [mat["nome"] for mat in st.session_state.materiais_db]
    
    default_index = 0
    try:
        default_index = next(i for i, nome in enumerate(material_nomes) if "Tijolo isolante" in nome)
    except StopIteration:
        pass # Mantém 0 se não encontrar

    nome_material_resistencia = st.selectbox(
        "Material da Parede de Resistência",
        material_nomes,
        index=default_index,
    )


# --- Preparação dos Objetos de Dados ---

# Coleta os objetos de dados com base nas entradas da barra lateral
try:
    dimensoes = DimensoesForno(
        altura_interna=altura_interna_mm / 1000.0,
        diametro_interno=diametro_interno_mm / 1000.0,
    )
    
    temperaturas = Temperaturas(
        ambiente=temp_ambiente_c + 273.15,
        lateral_interna=temp_interna_c + 273.15,
        parede_externa=temp_externa_c + 273.15,
    )
    
    material_resistencia = get_material_from_list(
        nome_material_resistencia, st.session_state.materiais_db
    )
    
    if not material_resistencia:
         st.sidebar.error("Material de resistência não encontrado. Verifique o banco de materiais.")
         st.stop()

except Exception as e:
    st.sidebar.error(f"Erro ao processar entradas: {e}")
    st.stop()


# --- Abas da Aplicação ---

tab_global, tab_especifica = st.tabs(["Análise Global", "Análise Específica"])


# --- TAB 1: Análise Global ---

with tab_global:
    st.header("Análise Global Comparativa")
    st.markdown(
        "Compare o desempenho de todos os materiais do banco de dados quando aplicados "
        "a todas as partes do forno (tampa, base e lateral)."
    )

    st.subheader("Banco de Materiais Isolantes")
    st.info(
        "Edite, adicione ou remova materiais na tabela abaixo. Os custos devem ser em R$/m³."
    )
    
    try:
        # --- INÍCIO DA CORREÇÃO DO BUG DO DATA_EDITOR ---
        edited_materials = st.data_editor(
            st.session_state.materiais_db,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "nome": st.column_config.TextColumn(
                    "Nome", required=True
                ),
                "k": st.column_config.NumberColumn(
                    "Condutividade (k)", 
                    min_value=0.001, 
                    step=0.001,  # Permite decimais pequenos
                    format="%.3f",
                    required=True # Evita erros em linhas novas
                ),
                "emissividade": st.column_config.NumberColumn(
                    "Emissividade (ε)", 
                    min_value=0.0, 
                    max_value=1.0, 
                    step=0.01, # Permite decimais
                    format="%.2f",
                    required=True # Evita erros em linhas novas
                ),
                "custo_por_m3": st.column_config.NumberColumn(
                    "Custo (R$/m³)", 
                    min_value=0.0, 
                    step=0.01, # Permite decimais
                    format="%.0f",
                    required=True # Evita erros em linhas novas
                ),
            }
        )
    except st.errors.StreamlitAPIException as e:
        st.error(f"Erro ao renderizar o editor de dados. Verifique a consistência dos dados. Detalhe: {e}")
        st.dataframe(st.session_state.materiais_db) # Mostra os dados brutos para debug
        st.stop()

    
    if edited_materials != st.session_state.materiais_db:
        st.session_state.materiais_db = edited_materials
        st.rerun() 
    
    if st.button("Executar Análise Global", type="primary", use_container_width=True):
        
        lista_de_resultados = []
        
        try:
            materiais_para_analise = [Material(**mat_dict) for mat_dict in st.session_state.materiais_db]
        except TypeError as e:
            st.error(f"Erro ao converter materiais do banco de dados. Verifique os dados inseridos. Detalhe: {e}")
            st.stop()

        with st.spinner("Executando análise comparativa..."):
            progress_bar = st.progress(0.0, text="Iniciando análise...")
            total_materiais = len(materiais_para_analise)
            
            for i, material_isolante in enumerate(materiais_para_analise):
                
                progress_bar.progress((i + 1) / total_materiais, text=f"Analisando: {material_isolante.nome}")
                
                resultados = executar_analise_completa(
                    material_tampa=material_isolante,
                    material_base=material_isolante,
                    material_lateral=material_isolante,
                    material_resistencia=material_resistencia,
                    dimensoes=dimensoes,
                    temperaturas=temperaturas,
                    espessura_parede_resistencia_mm=espessura_resistencia_mm,
                )

                if resultados:
                    lista_de_resultados.append(resultados)

        if lista_de_resultados:
            st.success("Análise Global Concluída!")
            
            df_resultados = pd.DataFrame(lista_de_resultados).set_index("Material")
            
            colunas_para_mostrar = [
                "e_lateral_mm", "e_base_mm", "e_tampa_mm", 
                "Potencia_Total", 
                "Custo_Lateral", "Custo_Base", "Custo_Tampa", "Custo_Total"
            ]
            df_display = df_resultados[colunas_para_mostrar].copy()
            df_display.columns = [
                "Esp. Lateral (mm)", "Esp. Base (mm)", "Esp. Tampa (mm)",
                "Potência (W)",
                "Custo Lat. (R$)", "Custo Base (R$)", "Custo Tampa (R$)", "Custo Total (R$)"
            ]
            
            st.subheader("Resultados Comparativos")
            st.dataframe(df_display.style.format("{:.0f}"))

            st.subheader("Gráficos Comparativos")
            
            fig_espessuras = plotar_grafico_espessuras(df_resultados.reset_index())
            st.pyplot(fig_espessuras)

            fig_custo_potencia = plotar_grafico_custo_potencia(df_resultados.reset_index())
            st.pyplot(fig_custo_potencia)
            
            st.session_state.global_results_df = df_resultados
            
        else:
            st.warning("Nenhuma análise foi concluída com sucesso.")

    if "global_results_df" in st.session_state:
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=True).encode('utf-8')

        csv = convert_df_to_csv(st.session_state.global_results_df)
        st.download_button(
            label="Download dos Resultados (CSV)",
            data=csv,
            file_name="analise_global_forno.csv",
            mime="text/csv",
        )


# --- TAB 2: Análise Específica ---

with tab_especifica:
    st.header("Análise Específica por Componente")
    st.markdown(
        "Selecione materiais diferentes para cada parte do forno e calcule "
        "o resultado para essa configuração específica."
    )

    material_nomes_especificos = [mat["nome"] for mat in st.session_state.materiais_db]

    st.subheader("Seleção de Materiais Específicos")
    
    col_tampa, col_base, col_lateral = st.columns(3)
    
    with col_tampa:
        nome_mat_tampa = st.selectbox(
            "Material da Tampa", material_nomes_especificos, key="mat_tampa"
        )
    with col_base:
        nome_mat_base = st.selectbox(
            "Material da Base", material_nomes_especificos, key="mat_base"
        )
    with col_lateral:
        nome_mat_lateral = st.selectbox(
            "Material da Lateral", material_nomes_especificos, key="mat_lateral"
        )

    if st.button("Executar Análise Específica", type="primary", use_container_width=True):
        
        mat_tampa = get_material_from_list(nome_mat_tampa, st.session_state.materiais_db)
        mat_base = get_material_from_list(nome_mat_base, st.session_state.materiais_db)
        mat_lateral = get_material_from_list(nome_mat_lateral, st.session_state.materiais_db)

        if not all([mat_tampa, mat_base, mat_lateral]):
            st.error("Erro ao carregar um dos materiais selecionados. Verifique o banco de materiais.")
            st.stop()

        with st.spinner("Executando análise específica..."):
            resultado_dict = executar_analise_completa(
                material_tampa=mat_tampa,
                material_base=mat_base,
                material_lateral=mat_lateral,
                material_resistencia=material_resistencia,
                dimensoes=dimensoes,
                temperaturas=temperaturas,
                espessura_parede_resistencia_mm=espessura_resistencia_mm,
            )

        if resultado_dict:
            st.success("Análise Específica Concluída!")
            st.session_state.specific_results = resultado_dict
            # Limpa dados antigos de sensibilidade se houver
            if "sensibilidade_df" in st.session_state:
                del st.session_state.sensibilidade_df
        else:
            st.error("A análise específica falhou.")

    # --- PONTO DO ERRO ---
    if "specific_results" in st.session_state:
        resultados = st.session_state.specific_results

        st.subheader("Resultados da Simulação")

        col1, col2, col3 = st.columns(3)
        col1.metric("Potência Total", f"{resultados['Potencia_Total']:.0f} W")
        col2.metric("Custo Total", f"R$ {resultados['Custo_Total']:.0f}")
        #col3.metric("Temp. Intermediária", f"{resultados['T_intermediaria_C']:.2f} °C")
        
        st.divider()

        col_e1, col_e2, col_e3 = st.columns(3)
        col_e1.metric("Espessura Lateral", f"{resultados['e_lateral_mm']:.0f} mm")
        col_e2.metric("Espessura Base", f"{resultados['e_base_mm']:.0f} mm")
        col_e3.metric("Espessura Tampa", f"{resultados['e_tampa_mm']:.0f} mm")

        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric("Custo Lateral", f"R$ {resultados['Custo_Lateral']:.0f}")
        col_c2.metric("Custo Base", f"R$ {resultados['Custo_Base']:.0f}")
        col_c3.metric("Custo Tampa", f"R$ {resultados['Custo_Tampa']:.0f}")

        st.divider()
        
        st.subheader("Correntes Elétricas Requeridas")
        correntes = calcular_correntes_eletricas(resultados['Potencia_Total'])
        
        col_i1, col_i2 = st.columns(2)
        col_i1.metric("Corrente (220V)", f"{correntes.corrente_220V:.1f} A")
        col_i2.metric("Corrente (110V)", f"{correntes.corrente_110V:.1f} A")

        # --- INÍCIO DO NOVO CÓDIGO  ---
        
        st.divider()
        st.subheader("Análise de Sensibilidade da Temperatura Ambiente")
        st.markdown(
            "Veja como a **potência total** necessária varia com a "
            "mudança da temperatura ambiente, mantendo as espessuras do forno fixas."
        )

        temp_range_c = st.slider(
            "Faixa de Temperatura Ambiente para Análise (°C)",
            min_value=-10,
            max_value=50,
            value=(5, 35),
            step=1
        )
        
        num_pontos = 20

        if st.button("Gerar Gráfico de Sensibilidade", use_container_width=True):
            
            mat_tampa = get_material_from_list(resultados['Mat_Tampa'], st.session_state.materiais_db)
            mat_base = get_material_from_list(resultados['Mat_Base'], st.session_state.materiais_db)
            mat_lateral = get_material_from_list(resultados['Mat_Lateral'], st.session_state.materiais_db)
            
            T_interna_K = temperaturas.lateral_interna
            T_externa_K = temperaturas.parede_externa
            
            espessuras_fixas = (
                resultados['e_tampa_mm'] / 1000.0,
                resultados['e_base_mm'] / 1000.0,
                resultados['e_lateral_mm'] / 1000.0,
                espessura_resistencia_mm / 1000.0
            )
            
            materiais = (mat_tampa, mat_base, mat_lateral, material_resistencia)

            T_amb_list_C = np.linspace(temp_range_c[0], temp_range_c[1], num_pontos)
            potencia_list_W = []

            with st.spinner(f"Calculando {num_pontos} pontos de potência..."):
                for T_amb_C in T_amb_list_C:
                    potencia = calcular_potencia_sensibilidade(
                        T_amb_C,
                        dimensoes,
                        T_interna_K,
                        T_externa_K,
                        materiais,
                        espessuras_fixas
                    )
                    if potencia:
                        potencia_list_W.append(potencia)
                    else:
                        potencia_list_W.append(np.nan)

            if any(potencia_list_W):
                fig_sensibilidade = plotar_grafico_sensibilidade_temp(T_amb_list_C, potencia_list_W)
                st.pyplot(fig_sensibilidade)
                
                df_sensibilidade = pd.DataFrame({
                    "Temperatura_Ambiente_C": T_amb_list_C,
                    "Potencia_Requerida_W": potencia_list_W
                })
                st.session_state.sensibilidade_df = df_sensibilidade
            else:
                st.error("Não foi possível calcular a análise de sensibilidade.")

        if "sensibilidade_df" in st.session_state:
            @st.cache_data
            def convert_sensibilidade_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_sensibilidade = convert_sensibilidade_to_csv(st.session_state.sensibilidade_df)
            st.download_button(
                label="Download Dados de Sensibilidade (CSV)",
                data=csv_sensibilidade,
                file_name="analise_sensibilidade_temp.csv",
                mime="text/csv",
                key="download_sens"
            )

        if st.button("Limpar Resultados Específicos"):
            del st.session_state.specific_results
            if "sensibilidade_df" in st.session_state:
                del st.session_state.sensibilidade_df
            st.rerun()