# CI/CD (Linux)

Le pipeline GitHub Actions est défini dans `.github/workflows/ci.yml`.

## Jobs

## 1) `quick-linux` (automatique sur push/PR)

Objectif: vérifier rapidement que le projet reste exécutable sans télécharger 10+ Go d'assets.

Vérifications:
- bootstrap Conda + installation des dépendances,
- validation syntaxe scripts shell,
- compilation Python (`compileall`),
- test de chargement config,
- vérification que le message d'erreur "assets manquants" est explicite quand `DOWNLOAD_ASSETS=0`.

## 2) `full-assets-linux` (manuel via `workflow_dispatch`)

Objectif: valider le runtime complet avec téléchargement des assets RoboCasa.

Vérifications:
- setup complet avec `DOWNLOAD_ASSETS=1` et validation d'assets,
- lancement de `python -m robocasa_telecom.sanity ...`.

Ce job est plus lent et plus coûteux (download volumineux), donc déclenché manuellement.

## Variables de setup importantes

- `DOWNLOAD_ASSETS=1`: télécharge les assets RoboCasa nécessaires au runtime.
- `VERIFY_ASSETS=1`: vérifie la présence d'assets critiques après téléchargement.
- `RUN_SETUP_MACROS=1`: initialise les macros RoboCasa/Robosuite.

## Commande locale Linux recommandée (setup complet)

```bash
ENV_NAME=robocasa_telecom \
DOWNLOAD_ASSETS=1 \
VERIFY_ASSETS=1 \
RUN_SETUP_MACROS=1 \
DOWNLOAD_DATASETS=0 \
bash scripts/setup_conda.sh
```
