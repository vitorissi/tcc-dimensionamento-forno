from .data_types import Material

MATERIAIS_DISPONIVEIS = [
    Material(nome="Placa Cerâmica 320 kg/m3", k=0.125, emissividade=0.90, custo_por_m3=13887), 
    Material(nome="Manta Cerâmica 64 kg/m3", k=0.175, emissividade=0.90, custo_por_m3=2303), 
    Material(nome="Manta Cerâmica 96 kg/m3", k=0.150, emissividade=0.90, custo_por_m3=3148), 
    Material(nome="Manta Cerâmica 128 kg/m3", k=0.125, emissividade=0.90, custo_por_m3=3346), 
    Material(nome="Manta Cerâmica 160 kg/m3", k=0.100, emissividade=0.90, custo_por_m3=4301),
    Material(nome="Tijolo isolante 0,8 g/cm3", k=0.30, emissividade=0.85, custo_por_m3=16417),
]