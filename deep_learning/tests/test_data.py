"""Testes básicos da estrutura do projeto deep_learning."""

from module_deep_learning.config import FIGURES_DIR, MODELS_DIR, PROJ_ROOT


def test_raiz_do_projeto_aponta_para_deep_learning():
    # A raiz calculada no config.py deve ser a pasta que contém o pacote
    assert (PROJ_ROOT / "module_deep_learning").is_dir()


def test_pastas_de_saida_existem():
    # Modelos treinados e figuras geradas são salvos nessas pastas
    assert MODELS_DIR.is_dir()
    assert FIGURES_DIR.is_dir()
