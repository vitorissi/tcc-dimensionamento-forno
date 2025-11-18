import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import argparse 

# Importações do projeto
from .calc import (
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
from .data_types import *
from .materials import MATERIAIS_DISPONIVEIS


def plotar_grafico_espessuras(df: pd.DataFrame):
    """Gera o gráfico de barras comparativas de espessura."""
    materiais_nomes = df["Material"].tolist()
    espessuras_lateral = df["e_lateral_mm"]
    espessuras_base = df["e_base_mm"]
    espessuras_tampa = df["e_tampa_mm"]

    componentes = ["Lateral", "Base", "Tampa"]
    x = np.arange(len(componentes))
    largura = 0.15

    plt.figure(figsize=(12, 6))

    for i, material in enumerate(materiais_nomes):
        plt.bar(
            x + i * largura,
            [espessuras_lateral.iloc[i], espessuras_base.iloc[i], espessuras_tampa.iloc[i]],
            largura,
            label=material,
        )

    plt.xticks(x + (len(materiais_nomes) - 1) * largura / 2, componentes)
    plt.ylabel("Espessura (mm)")
    plt.xlabel("Componente")
    plt.title("Comparativo de Espessura dos Isolantes")
    plt.legend(title="Material", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("comparativo_espessuras.png")
    print("Gráfico 'comparativo_espessuras.png' salvo.")


def plotar_grafico_custo_potencia(df: pd.DataFrame):
    """Gera o gráfico de dispersão Custo Total x Potência Total."""
    colors = ["blue", "red", "green", "purple", "orange", "brown"]
    markers = ["o", "s", "^", "D", "P", "*"]

    plt.figure(figsize=(10, 6))

    for i, mat in df.iterrows():
        print(
            f"Material: {mat['Material']}, Custo Total: {mat['Custo_Total']:.2f} R$, Potência Total: {mat['Potencia_Total']:.2f} W"
        )
        plt.scatter(
            mat["Custo_Total"],
            mat["Potencia_Total"],
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=mat["Material"],
            s=100,
        )

    plt.xlabel("Custo Total (R$)")
    plt.ylabel("Potência Total Requerida (W)")
    plt.title("Potência Requerida vs. Custo Total do Isolamento")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    formatter = mticker.ScalarFormatter(useOffset=False, useMathText=False)
    formatter.set_scientific(False)
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.savefig("custo_vs_potencia.png")
    print("Gráfico 'custo_vs_potencia.png' salvo.")


def executar_analise_completa(
    material_tampa: Material,       
    material_base: Material,       
    material_lateral: Material,     
    material_resistencia: Material,
    dimensoes: DimensoesForno,
    temperaturas: Temperaturas
) -> dict | None:
    """
    Executa a simulação térmica completa para um conjunto de materiais.
    Retorna um dicionário com os resultados, ou None se falhar.
    """
    
    # Gera um nome "combinado" para o caso da análise em lote
    nomes_materiais = set(
        [material_tampa.nome, material_base.nome, material_lateral.nome]
    )
    if len(nomes_materiais) == 1:
        nome_analise = material_tampa.nome
    else:
        nome_analise = f"Lateral({material_lateral.nome})/Tampa({material_tampa.nome})/Base({material_base.nome})"
    
    print(f"\n--- Analisando Combinação: {nome_analise} ---")
    
    # --- Início da lógica de cálculo ---
    propriedades_ar = propriedades_do_ar(temperaturas)
    h_ext_base, h_ext_tampa = coeficiente_conveccao_natural( 
    dimensoes, temperaturas, propriedades_ar
    )  

    espessura_parede_resistencia = 63 * (10**-3)

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
        print(f"Análise de radiação falhou para {nome_analise}. Pulando.")
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
    espessura_isolamento_lateral_inicial = 100 * 10**-3 
    
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
        print(f"Análise lateral falhou para {nome_analise}. Pulando.")
        return None
        
    # --- Cálculo de Custo  ---
    vol_tampa = calc_volume_parede_plana(dimensoes.area_tampa, espessura_tampa)
    custo_tampa = calc_custo_total_material(vol_tampa, material_tampa.custo_por_m3)
    
    vol_base = calc_volume_parede_plana(dimensoes.area_base, espessura_base)
    custo_base = calc_custo_total_material(vol_base, material_base.custo_por_m3)
    
    vol_lateral = calc_volume_parede_cilindrica(
        raio_interno=raio_intermediario,
        raio_externo=resultados_laterais.raio_externo_final,
        altura=dimensoes.altura_interna
    )
    custo_lateral = calc_custo_total_material(vol_lateral, material_lateral.custo_por_m3) # <-- MUDANÇA
    
    custo_total = custo_tampa + custo_base + custo_lateral
    
    # --- Coleta de Resultados ---
    return {
        "Material": nome_analise,
        "e_lateral_m": resultados_laterais.espessura_isolamento_lateral,
        "e_base_m": espessura_base,
        "e_tampa_m": espessura_tampa,
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

def get_material_by_name(nome: str, lista_materiais: list) -> Material | None:
    """Encontra um objeto Material em uma lista pelo seu nome."""
    for mat in lista_materiais:
        if mat.nome == nome:
            return mat
    return None

def main():
    # --- Inputs Globais ---
    dimensoes = DimensoesForno(
        altura_interna=450 * (10**-3), diametro_interno=400 * (10**-3)
    )

    temperaturas = Temperaturas(
        ambiente=25 + 273.15, lateral_interna=1250 + 273.15, parede_externa=75 + 273.15
    )

    material_resistencia = Material(
        nome="Tijolo isolante 0.8", k=0.3, emissividade=0.85, custo_por_m3=0.0
    )
    
    # --- LÓGICA DE EXECUÇÃO ---
    # Por padrão, rodamos a ANÁLISE EM LOTE.
    # O Streamlit irá importar e chamar `executar_analise_completa` diretamente.
    
    print("--- MODO DE ANÁLISE EM LOTE (COMPARATIVO) ---")
    lista_de_resultados = []

    for material_isolante in MATERIAIS_DISPONIVEIS:
            
        # Aqui chamamos a função passando o MESMO material para as 3 partes
        resultados = executar_analise_completa(
            material_tampa=material_isolante,   
            material_base=material_isolante,     
            material_lateral=material_isolante, 
            material_resistencia=material_resistencia,
            dimensoes=dimensoes,
            temperaturas=temperaturas
        )
        
        if resultados:
            lista_de_resultados.append(resultados)

    if not lista_de_resultados:
        print("Nenhuma simulação em lote foi concluída. Encerrando.")
        return

    df_resultados = pd.DataFrame(lista_de_resultados)
    #print("\n--- DataFrame de Resultados ---")
    #print(df_resultados.to_string())

    plotar_grafico_espessuras(df_resultados)
    plotar_grafico_custo_potencia(df_resultados)
    print("\nAnálise em lote concluída. Gráficos salvos.")


if __name__ == "__main__":
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError as e:
        print(f"Erro: Dependência não encontrada: {e.name}")
        print("Por favor, instale as dependências com: pip install pandas matplotlib")
        exit(1)
        
    main()