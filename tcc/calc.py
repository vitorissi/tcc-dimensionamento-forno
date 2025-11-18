import numpy as np
from CoolProp.CoolProp import PropsSI
from .consts import P, G, SIGMA
from .data_types import (
    PropriedadesAr,
    ResultadosTermicosLaterais,
    DimensoesExternas,
    CorrentesEletricas,
    DimensoesForno, 
)


def propriedades_do_ar(temperaturas):
    """Processa os dados do forno e retorna propriedades calculadas."""
    Tf = temperaturas.Tf

    # Propriedades do ar
    mu = PropsSI("V", "T", Tf, "P", P, "Air")  # Viscosidade dinâmica
    rho = PropsSI("D", "T", Tf, "P", P, "Air")  # Densidade
    k_ar = PropsSI("L", "T", Tf, "P", P, "Air")  # Condutividade térmica
    cp = PropsSI("C", "T", Tf, "P", P, "Air")  # Calor específico

    nu = mu / rho
    alpha = k_ar / (rho * cp)
    beta = 1 / Tf
    Pr = nu / alpha

    return PropriedadesAr(
        mu=mu, rho=rho, k_ar=k_ar, cp=cp, nu=nu, alpha=alpha, beta=beta, Pr=Pr
    )


def coeficiente_conveccao_natural(dimensoes_forno, temperaturas, propriedades_ar):
    """Analise da convecção natural do exterior da base e da tampa."""
    perimetro_tampa = 2 * np.pi * dimensoes_forno.raio_interno
    comprimento_equivalente = dimensoes_forno.area_tampa / perimetro_tampa

    beta = propriedades_ar.beta
    nu = propriedades_ar.nu
    alpha = propriedades_ar.alpha
    delta_t = temperaturas.parede_externa - temperaturas.ambiente
    ra_tampa = (G * beta * delta_t * comprimento_equivalente**3) / (nu * alpha)

    # Parede Base (superficie inferior aquecida)
    nusselt_base = 0.27 * ra_tampa ** (1 / 4)
    h_ext_base = nusselt_base * (propriedades_ar.k_ar / comprimento_equivalente)

    # Parede Tampa (superfície superior placa aquecida)
    nusselt_tampa = 0.54 * ra_tampa ** (1 / 4)
    h_ext_tampa = nusselt_tampa * (propriedades_ar.k_ar / comprimento_equivalente)

    return h_ext_base, h_ext_tampa


def taxa_perda_sup_plana(area, h_ext, emissividade, temperaturas, propriedades_ar):
    h_r = (
        emissividade
        * SIGMA
        * (
            (temperaturas.parede_externa + temperaturas.ambiente)
            * (temperaturas.parede_externa**2 + temperaturas.ambiente**2)
        )
    )

    # Radiação na parede externa
    taxa_radiacao_ext = (
        area
        * emissividade
        * SIGMA
        * (temperaturas.parede_externa**4 - temperaturas.ambiente**4)
    )

    # Convecção na parede externa
    taxa_conveccao_ext = (
        area * h_ext * (temperaturas.parede_externa - temperaturas.ambiente)
    )

    # Balanço de energia
    taxa_radiacao_int = taxa_radiacao_ext + taxa_conveccao_ext

    return taxa_radiacao_int


def fator_de_forma(dimensoes_forno):
    s = 2 + (dimensoes_forno.altura_interna / dimensoes_forno.raio_interno) ** 2
    f_bt = (s - (s**2 - 4) ** 0.5) / 2
    f_bl = 1 - f_bt
    return f_bl, f_bt


def troca_radiacao_superficies(
    temperaturas,
    emis_base_interna,
    emis_tampa_interna,
    emis_lateral_interna,
    dimensoes_forno,
    f_bl,
    f_bt,
    taxa_radiacao_int_base,
    taxa_radiacao_int_tampa,
    max_iter=1000,
    erro_adm=1e-1,
):
    """
    Calcula a troca de radiação entre as superfícies do forno utilizando fatores de forma.
    """
    # Suposições iniciais
    t_base_interna = t_tampa_interna = temperaturas.lateral_interna - 50

    for i in range(max_iter):
        # Definindo os coeficientes das equações
        a11 = -f_bl
        a12 = 2 * f_bl + (
            (emis_lateral_interna * dimensoes_forno.area_lateral_interna)
            / ((1 - emis_lateral_interna) * dimensoes_forno.area_base)
        )
        a13 = -f_bl
        a21 = f_bl + f_bt + ((emis_base_interna) / (1 - emis_base_interna))
        a22 = -f_bl
        a23 = -f_bt
        a31 = -f_bt
        a32 = -f_bl
        a33 = f_bl + f_bt + ((emis_tampa_interna) / (1 - emis_tampa_interna))

        # Definindo os termos independentes
        b1 = (
            SIGMA
            * (temperaturas.lateral_interna**4)
            * emis_lateral_interna
            * dimensoes_forno.area_lateral_interna
        ) / (dimensoes_forno.area_base * (1 - emis_lateral_interna))
        b2 = (SIGMA * (t_base_interna**4) * emis_base_interna) / (1 - emis_base_interna)
        b3 = (SIGMA * (t_tampa_interna**4) * emis_tampa_interna) / (
            1 - emis_tampa_interna
        )

        A = np.array(
            [
                [
                    a11,
                    a12,
                    a13,
                ],  # Representando a matriz de coeficientes (A) usando as variáveis
                [a21, a22, a23],
                [a31, a32, a33],
            ]
        )

        B = np.array(
            [b1, b2, b3]
        )  # Representando o vetor de termos independentes (B) usando as variáveis

        J = np.linalg.solve(A, B)  # Resolvendo o sistema

        J_b, J_l, J_t = J

        # calcular novas temperaturas com base nas constantes obtidas
        t_base_interna_novo = (
            (J_b / SIGMA)
            + (
                (taxa_radiacao_int_base * (1 - emis_base_interna))
                / (emis_base_interna * dimensoes_forno.area_base)
            )
        ) ** (1 / 4)
        t_tampa_interna_novo = (
            (J_t / SIGMA)
            + (
                (taxa_radiacao_int_tampa * (1 - emis_tampa_interna))
                / (emis_tampa_interna * dimensoes_forno.area_tampa)
            )
        ) ** (1 / 4)

        # Verificar convergência
        if (
            abs(t_base_interna_novo - t_base_interna) < erro_adm
            and abs(t_tampa_interna_novo - t_tampa_interna) < erro_adm
        ):
            print(f"Convergiu em {i+1} iterações.")
            break

        # Atualiza para próxima iteração
        t_base_interna, t_tampa_interna = t_base_interna_novo, t_tampa_interna_novo

    else:
        print("Atenção: não convergiu após o número máximo de iterações.")
        return

    # Cálculo da taxa de radiação interna na parede lateral
    q_radiacao_lateral_interna = (
        emis_lateral_interna
        * dimensoes_forno.area_lateral_interna
        * ((SIGMA * temperaturas.lateral_interna**4) - J_l)
    ) / (1 - emis_lateral_interna)

    return t_base_interna, t_tampa_interna, q_radiacao_lateral_interna


def resistencia_total(t_interna, t_externa, taxa_conducao):
    return (t_interna - t_externa) / taxa_conducao


def calc_espessura(
    material, t_interna, t_parede_externa, taxa_conducao, area):
    return (material.k * area * (t_interna - t_parede_externa)) / taxa_conducao


def grashof(t_parede_externa, t_ambiente, altura_interna, beta, nu):
    return (G * beta * (t_parede_externa - t_ambiente) * altura_interna**3) / (nu**2)


def numero_de_raylength(gr_l, Pr):
    return gr_l * Pr


def nusselt_lateral(raylength_lateral, Pr):
    return (
        0.825
        + (
            (0.387 * (raylength_lateral ** (1 / 6)))
            / ((1 + (0.492 / Pr) ** (9 / 16)) ** (8 / 27))
        )
    ) ** 2


def coeficiente_conveccao_natural_lateral(nusselt_lateral, k_ar, altura_interna):
    return nusselt_lateral * (k_ar / altura_interna)


def calc_raio_intermediario(raio_interno, espessura_parede_resistencia):
    return raio_interno + espessura_parede_resistencia


def calc_volume_parede_resistencia(raio_intermediario, raio_interno, altura_interna):
    return np.pi * (raio_intermediario**2 - raio_interno**2) * altura_interna


# -----------------------------------------------
# FUNÇÕES DA ANÁLISE LATERAL REFATORADAS
# -----------------------------------------------


def _calcular_passo_analise_lateral(
    espessura_isolamento_atual,
    espessura_parede_resistencia,
    dimensoes_forno,
    material_lateral,
    material_resistencia,
    temperaturas,
    h_ext_lateral,
    q_radiacao_lateral_interna,
    volume_parede_resistencia,
    raio_intermediario,
):
    """Calcula um único passo da iteração da análise lateral."""

    # Geometria externa baseada na espessura atual
    raio_externo = (
        dimensoes_forno.raio_interno
        + espessura_parede_resistencia
        + espessura_isolamento_atual
    )
    area_lateral_externa = 2 * np.pi * raio_externo * dimensoes_forno.altura_interna

    # Perdas externas
    q_rad_ext_l = (
        area_lateral_externa
        * material_lateral.emissividade
        * SIGMA
        * (temperaturas.parede_externa**4 - temperaturas.ambiente**4)
    )
    q_conv_ext_l = (
        area_lateral_externa
        * h_ext_lateral
        * (temperaturas.parede_externa - temperaturas.ambiente)
    )
    q_cond_l = q_rad_ext_l + q_conv_ext_l

    # Calculo Energia gerada
    energia_gerada = abs(q_radiacao_lateral_interna) + q_cond_l  # balanco de energia
    q_gerada_vol = energia_gerada / volume_parede_resistencia

    # Temperatura camada intermediaria
    fluxo_perda_lateral = q_cond_l / (
        2 * np.pi * raio_intermediario * dimensoes_forno.altura_interna
    )

    c1 = (
        (q_gerada_vol * raio_intermediario / 2) - fluxo_perda_lateral
    ) * raio_intermediario / material_resistencia.k
    c2 = (
        temperaturas.lateral_interna
        + (q_gerada_vol / (4 * material_resistencia.k)) * dimensoes_forno.raio_interno**2
        - c1 * np.log(dimensoes_forno.raio_interno)
    )
    temperatura_intermediaria = (
        -(q_gerada_vol / (4 * material_resistencia.k)) * raio_intermediario**2
        + c1 * np.log(raio_intermediario)
        + c2
    )

    # Nova espessura
    resistencia_total_lateral = (
        temperatura_intermediaria - temperaturas.parede_externa
    ) / q_cond_l
    espessura_isolamento_lateral_nova = (
        np.e
        ** (
            resistencia_total_lateral
            * 2
            * np.pi
            * material_lateral.k
            * dimensoes_forno.altura_interna
        )
    ) * raio_intermediario - raio_intermediario

    # Retorna os valores calculados neste passo
    return (
        q_cond_l,
        energia_gerada,
        temperatura_intermediaria,
        espessura_isolamento_lateral_nova,
        raio_externo,
    )


def analise_lateral(
    espessura_parede_resistencia,
    temperaturas,
    dimensoes_forno,
    material_lateral,
    material_resistencia,
    q_radiacao_lateral_interna,
    h_ext_lateral,
    volume_parede_resistencia,
    raio_intermediario,
    chute_espessura_isolamento_lateral=0.100,  # Palpite inicial
    max_iter=1000,
    erro_adm=1e-4,
) -> ResultadosTermicosLaterais | None:
    """
    Executa a análise iterativa da parede lateral para encontrar a espessura
    do isolamento e a potência necessária.
    
    (Esta é a definição de função correta. Note que os argumentos
    com default, 'max_iter' e 'erro_adm', estão NO FINAL.)
    """

    espessura_atual = chute_espessura_isolamento_lateral

    for i in range(max_iter):
        # Chama a função auxiliar para calcular um passo
        (
            q_cond_l,
            energia_gerada,
            T_intermediaria,
            espessura_nova,
            raio_externo,
        ) = _calcular_passo_analise_lateral(
            espessura_atual,
            espessura_parede_resistencia,
            dimensoes_forno,
            material_lateral,
            material_resistencia,
            temperaturas,
            h_ext_lateral,
            q_radiacao_lateral_interna,
            volume_parede_resistencia,
            raio_intermediario,
        )

        # Verificar convergência
        if abs(espessura_nova - espessura_atual) < erro_adm:
            print(f"Análise lateral convergiu em {i+1} iterações.")

            # Roda a simulação mais uma vez com a espessura final
            (
                q_cond_l_final,
                energia_gerada_final,
                T_intermediaria_final,
                _,
                raio_externo_final,
            ) = _calcular_passo_analise_lateral(
                espessura_nova,  # <-- Usando o valor convergido
                espessura_parede_resistencia,
                dimensoes_forno,
                material_lateral,
                material_resistencia,
                temperaturas,
                h_ext_lateral,
                q_radiacao_lateral_interna,
                volume_parede_resistencia,
                raio_intermediario,
            )

            # Retorna o objeto de resultados TÉRMICOS
            return ResultadosTermicosLaterais(
                espessura_isolamento_lateral=espessura_nova,
                perda_termica_lateral=q_cond_l_final,
                temperatura_intermediaria=T_intermediaria_final,
                potencia_total_necessaria=energia_gerada_final,
                raio_externo_final=raio_externo_final,
            )

        # Atualiza para próxima iteração
        espessura_atual = espessura_nova

    else:
        print(f"Atenção: Análise lateral não convergiu após {max_iter} iterações.")
        return None


def calcular_dimensoes_externas(
    dimensoes_forno: DimensoesForno,
    resultados_laterais: ResultadosTermicosLaterais,
    espessura_base: float,
    espessura_tampa: float,
) -> DimensoesExternas:
    """Calcula as dimensões externas finais do forno."""

    altura_externa = (
        dimensoes_forno.altura_interna + espessura_base + espessura_tampa
    )
    raio_externo = resultados_laterais.raio_externo_final
    diametro_externo = raio_externo * 2

    return DimensoesExternas(
        altura_externa=altura_externa,
        raio_externo=raio_externo,
        diametro_externo=diametro_externo,
    )


def calcular_correntes_eletricas(
    potencia_necessaria: float,
) -> CorrentesEletricas:
    """Calcula a corrente elétrica para redes de 110V e 220V."""

    corrente_220 = potencia_necessaria / 220
    corrente_110 = potencia_necessaria / 110

    return CorrentesEletricas(
        corrente_220V=corrente_220, corrente_110V=corrente_110
    )

def calc_volume_parede_plana(area: float, espessura: float) -> float:
    """Calcula o volume de um isolamento de parede plana."""
    return area * espessura

def calc_volume_parede_cilindrica(
    raio_interno: float, raio_externo: float, altura: float
) -> float:
    """Calcula o volume de um isolamento de parede cilíndrica."""
    return np.pi * (raio_externo**2 - raio_interno**2) * altura

def calc_custo_total_material(
    volume: float, custo_por_m3: float
) -> float:
    """Calcula o custo total de um material dado seu volume e custo por m³."""
    return volume * custo_por_m3