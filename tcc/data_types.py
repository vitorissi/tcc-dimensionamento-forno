import numpy as np
from dataclasses import dataclass
from enum import Enum


@dataclass
class DimensoesForno:
    altura_interna: float  # [m]
    diametro_interno: float  # [m]

    @property
    def raio_interno(self) -> float:
        return self.diametro_interno / 2

    @property
    def volume_interno(self) -> float:
        return np.pi * self.raio_interno**2 * self.altura_interna

    @property
    def area_lateral_interna(self) -> float:
        return 2 * np.pi * self.raio_interno * self.altura_interna

    @property
    def area_tampa(self) -> float:
        return np.pi * self.raio_interno**2

    @property
    def area_base(self) -> float:
        return np.pi * self.raio_interno**2


@dataclass
class Temperaturas:
    ambiente: float  # [K]
    lateral_interna: float  # [K]
    parede_externa: float  # [K]

    @property
    def Tf(self) -> float:
        """Temperatura de filme (média entre ambiente e parede externa)."""
        return (self.parede_externa + self.ambiente) / 2


@dataclass
class PropriedadesAr:
    mu: float  # Viscosidade dinâmica [Pa.s]
    rho: float  # Densidade [kg/m³]
    k_ar: float  # Condutividade térmica [W/m·K]
    cp: float  # Calor específico [J/kg·K]
    nu: float  # Viscosidade cinemática [m²/s]
    alpha: float  # Difusividade térmica [m²/s]
    beta: float  # Coeficiente de expansão térmica [1/K]
    Pr: float  # Número de Prandtl


class ParteForno(Enum):
    LATERAL_EXT = "lateral_externa"
    LATERAL_INT = "lateral_interna"
    TAMPA = "tampa"
    BASE = "base"


@dataclass
class Material:
    nome: str
    k: float  # [W/mK]
    emissividade: float  
    custo_por_m3: float  # [R$/m³] 


@dataclass
class ParteFornoMaterial:
    parte: ParteForno
    material: Material

# Em data_types.py

@dataclass
class ResultadosTermicosLaterais:
    """Agrupa os resultados da análise térmica da parede lateral."""
    espessura_isolamento_lateral: float  # [m]
    perda_termica_lateral: float        # [W]
    temperatura_intermediaria: float    # [K]
    potencia_total_necessaria: float    # [W]
    raio_externo_final: float           # [m]

@dataclass
class DimensoesExternas:
    """Agrupa as dimensões externas finais do forno."""
    altura_externa: float   # [m]
    raio_externo: float     # [m]
    diametro_externo: float # [m]

@dataclass
class CorrentesEletricas:
    """Agrupa as correntes calculadas para diferentes tensões."""
    corrente_220V: float  # [A]
    corrente_110V: float  # [A]

