"""Testes básicos da estrutura do projeto deep_learning."""

from module_deep_learning.config import FIGURES_DIR, MODELS_DIR, PROJ_ROOT


def test_raiz_do_projeto_aponta_para_deep_learning():
    # A raiz calculada no config.py deve ser a pasta que contém o pacote
    assert (PROJ_ROOT / "module_deep_learning").is_dir()


def test_pastas_de_saida_existem():
    # Modelos treinados e figuras geradas são salvos nessas pastas
    assert MODELS_DIR.is_dir()
    assert FIGURES_DIR.is_dir()


def _pacientes_exemplo():
    """Três pacientes fictícios com as mesmas colunas do heart.csv."""
    import pandas as pd

    return pd.DataFrame(
        {
            "age": [63, 45, 58],
            "sex": [1, 0, 1],
            "cp": [1, 2, 4],
            "trestbps": [145, 130, 120],
            "chol": [233, 250, 300],
            "fbs": [1, 0, 0],
            "restecg": [2, 0, 1],
            "thalach": [150, 170, 120],
            "exang": [0, 0, 1],
            "oldpeak": [2.3, 0.5, 1.8],
            "slope": [3, 1, 2],
            "ca": [0, 0, 2],
            "thal": ["fixed", "normal", "reversible"],
            "target": [0, 0, 1],
        }
    )


def test_features_normaliza_numericas_e_gera_one_hot():
    from module_deep_learning.features import encode_features, fit_preprocessing

    pacientes = _pacientes_exemplo()
    params = fit_preprocessing(pacientes)
    features = encode_features(pacientes, params)

    # Numérica normalizada com as estatísticas do próprio treino tem média ~0
    assert abs(features["age"].mean()) < 1e-6
    # Cada categoria vira uma coluna 0/1 e cada paciente tem exatamente um 1 por feature
    assert {"thal_fixed", "thal_normal", "thal_reversible"} <= set(features.columns)
    assert (features[["thal_fixed", "thal_normal", "thal_reversible"]].sum(axis=1) == 1).all()


def test_categoria_desconhecida_vira_vetor_de_zeros():
    from module_deep_learning.features import encode_features, fit_preprocessing

    pacientes = _pacientes_exemplo()
    params = fit_preprocessing(pacientes)
    novo = pacientes.head(1).assign(thal="desconhecido")

    features = encode_features(novo, params)
    assert features.filter(like="thal_").sum(axis=1).iloc[0] == 0


def test_modelo_devolve_uma_probabilidade_por_paciente():
    from module_deep_learning.features import encode_features, fit_preprocessing
    from module_deep_learning.modeling.train import create_model

    pacientes = _pacientes_exemplo()
    features = encode_features(pacientes, fit_preprocessing(pacientes))
    model = create_model(features.shape[1])

    probabilidades = model.predict(features.to_numpy("float32"), verbose=0)
    assert probabilidades.shape == (3, 1)
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()
