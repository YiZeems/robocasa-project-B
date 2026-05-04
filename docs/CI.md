# CI/CD

Le pipeline GitHub Actions doit maintenant s'appuyer sur `uv` plutôt que sur Conda.

## Jobs

## 1) `quick-linux`

Vérifications recommandées:
- `uv sync`
- `shellcheck` / validation syntaxe scripts shell
- `python -m compileall`
- test de chargement de config
- vérification que le message d'erreur "assets manquants" reste explicite quand `DOWNLOAD_ASSETS=0`

## 2) `full-assets-linux`

Vérifications recommandées:
- `DOWNLOAD_ASSETS=1`
- `VERIFY_ASSETS=1`
- lancement de `uv run python -m robocasa_telecom.sanity ...`

## Setup local recommandé

```bash
DOWNLOAD_ASSETS=1 \
VERIFY_ASSETS=1 \
RUN_SETUP_MACROS=1 \
DOWNLOAD_DATASETS=0 \
bash scripts/setup_uv.sh
```
