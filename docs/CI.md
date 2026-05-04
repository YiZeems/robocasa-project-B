# CI/CD

Le pipeline GitHub Actions doit maintenant s'appuyer sur `uv` plutôt que sur Conda.

## Jobs

## 1) `quick-linux`

Vérifications recommandées:
- `uv sync`
- `shellcheck` / validation syntaxe scripts shell
- `uv pip check`
- `uv run python -m compileall robocasa_telecom tests`
- test de chargement de config
- vérification que le message d'erreur "assets manquants" reste explicite quand `DOWNLOAD_ASSETS=0`

## 2) `full-assets-linux`

Vérifications recommandées:
- `DOWNLOAD_ASSETS=1`
- `VERIFY_ASSETS=1`
- `RUN_SETUP_MACROS=1`
- lancement de `uv run python -m robocasa_telecom.sanity ...`
- smoke train court: `uv run python -m robocasa_telecom.train --config configs/train/open_single_door_sac_debug.yaml --seed 0 --total-timesteps 10 --no-auto-resume`

## Setup local recommandé

```bash
DOWNLOAD_ASSETS=1 \
VERIFY_ASSETS=1 \
RUN_SETUP_MACROS=1 \
DOWNLOAD_DATASETS=0 \
bash scripts/setup_uv.sh
```
