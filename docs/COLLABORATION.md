# Collaboration Git (équipe de 4)

Branches existantes:
- `main`: base stable à exécuter
- `develop`: intégration continue de l'équipe
- `feature/member1-env-foundation`
- `feature/member2-training-pipeline`
- `feature/member3-eval-visualization`
- `feature/member4-slurm-repro`

## Règles recommandées

1. Chaque membre travaille sur sa branche `feature/...`.
2. Ouvrir PR vers `develop`.
3. Après validation, merge vers `develop`.
4. Quand `develop` est stable, fast-forward/merge vers `main`.

## Convention commits

- Préfixe type: `env:`, `rl:`, `docs:`, `slurm:`
- Message court et actionnable.

Exemple:
- `env: add fallback for missing task aliases`
- `rl: export training curve csv`
- `docs: update runbook with SLURM examples`
