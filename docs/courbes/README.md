# Courbes d'entraînement - RoboCasa OpenCabinet

Ce dossier rassemble les figures utilisées pour analyser les runs SAC sur la tâche d'ouverture de porte. Les courbes ne servent pas seulement à illustrer les résultats : elles expliquent pourquoi chaque configuration a été arrêtée ou modifiée.

Pour la discussion complète, voir le [rapport de projet](../project_report.md).

## Runs comparés

| Couleur | Run | Changement principal | Steps analysés |
|---|---|---|---:|
| Rouge | SAC v1 | `ent_coef="auto"` | 400k sur 500k |
| Orange | SAC v2 | `ent_coef="auto_0.1"`, `target_entropy=-4` | 400k sur 900k |
| Bleu | SAC v3 | `ent_coef=0.1` fixe, SDE | 400k |
| Vert | SAC v3 curriculum | seuil plus facile, spawn réduit | 400k sur 500k |
| Turquoise | SAC HER v2 | HER + reward dense hybride | 300k |

HER v1 n'est pas affiché dans les courbes combinées : en reward sparse pure, les métriques restent plates et n'ajoutent pas d'information visuelle.

## Angle maximal de la porte

![val_door_angle_max_mean](combined/combined_val_door_angle_max.png)

Cette métrique mesure l'angle maximal atteint pendant un épisode de validation. Elle est plus informative que le return, car elle décrit directement l'effet physique de la politique sur la porte.

- v1 et v2 restent autour de 0.014-0.017 rad : la porte ne s'ouvre pas.
- v3 est plus stable en entraînement, mais n'améliore pas l'ouverture.
- Le curriculum monte brièvement à environ 0.039 rad, sans convergence.
- HER v2 atteint 0.133 rad à 200k steps : c'est le premier vrai mouvement de porte observé.

Le succès strict reste à 0 %, mais HER v2 est clairement le meilleur signal expérimental du projet.

## Coefficient d'entropie SAC

![train_ent_coef](combined/combined_train_ent_coef.png)

Le coefficient d'entropie `alpha` contrôle l'exploration dans SAC. Les deux premiers runs montrent le problème principal : avec l'auto-tuning, `alpha` chute vers zéro. La politique devient presque déterministe avant d'avoir découvert la stratégie de contact.

À partir de v3, `ent_coef=0.1` est fixé manuellement. La courbe devient plate et l'entraînement cesse d'exploser. Cette correction stabilise SAC, même si elle ne suffit pas à résoudre la tâche.

## Critic loss

![train_critic_loss](combined/combined_train_critic_loss.png)

La critic loss mesure l'erreur des Q-values. Sur v1 et v2, elle atteint des valeurs pathologiques, jusqu'à plusieurs dizaines de milliers. Cela confirme que l'actor reçoit des gradients inutilisables.

À partir de v3, la critic loss reste contrôlée. HER v2 montre un pic initial raisonnable, puis revient vers une zone stable. Cette courbe confirme que le problème final n'est plus une divergence numérique, mais le manque de signal comportemental.

## Return de validation

![val_return_mean](combined/combined_val_return_mean.png)

Le return doit être interprété avec prudence, car les rewards ne sont pas exactement comparables entre les runs. HER v2 utilise notamment une composante sparse négative, ce qui change l'échelle.

La tendance reste utile :

- v1, v2 et v3 stagnent.
- Le curriculum produit un pic ponctuel, puis régresse.
- HER v2 s'améliore de façon plus régulière, même sans succès strict.

Le return ne suffit donc pas à conclure, mais il soutient l'idée que HER v2 apprend quelque chose de plus structuré.

## Fraction de reward d'approche

![val_approach_frac_mean](combined/combined_val_approach_frac.png)

`approach_frac` indique la part de reward provenant du simple fait d'être proche de la poignée. Une valeur élevée signale du hover-hacking : la politique reste près de la poignée au lieu d'ouvrir la porte.

HER v2 est sain autour de 200k steps (`approach_frac` faible), puis commence à dériver à 300k. Cette dérive correspond à la baisse de `door_angle_max`. Le diagnostic est donc cohérent : l'agent a appris à approcher, mais pas encore à pousser et maintenir l'ouverture.

## Actor loss

![train_actor_loss](combined/combined_train_actor_loss.png)

Dans SAC, une actor loss qui devient plus négative indique généralement que les Q-values estimées augmentent pour les actions choisies. Une remontée vers zéro ou vers le positif est un signe d'alerte.

- v1 et v2 remontent fortement : entraînement cassé.
- v3 et curriculum restent négatifs et stables : SAC est sain mais bloqué.
- HER v2 descend jusqu'à environ 200k, puis remonte légèrement : début de dérive vers le hover-hacking.

Cette courbe sert de go/no-go : si l'actor loss remonte en positif, le run ne mérite pas d'être prolongé.

## Synthèse des diagnostics

```text
SAC v1
  ent_coef auto -> alpha vers 0 -> critic loss énorme -> aucun apprentissage

SAC v2
  auto-tuning modifié, mais même effondrement de l'entropie

SAC v3
  entropie fixe, entraînement stable, mais cold start : pas de contact utile

SAC v3 curriculum
  léger progrès, puis régression ; le replay buffer reste presque sans signal positif

SAC HER v1
  reward sparse pure, pas assez de guidage vers la poignée

SAC HER v2
  meilleur résultat : 0.133 rad à 200k steps, puis début de hover-hacking
```

## Tableau final

| Métrique | Objectif | v1 | v2 | v3 | Curriculum | HER v2 @200k |
|---|---:|---:|---:|---:|---:|---:|
| `door_angle_max` | > 0.10 rad | 0.014 | 0.017 | 0.012 | 0.039 | **0.133** |
| `ent_coef` | stable | crash | crash | 0.1 | 0.1 | 0.1 |
| `critic_loss` | contrôlée | 48k+ | 116k+ | ~35 | ~10 | ~5 |
| `val_return_mean` | croissant | stagne | stagne | stagne | pic ponctuel | **croissant** |
| `approach_frac` | faible | élevé | élevé | moyen | moyen | **faible** |
| `val_success_rate` | > 0 % | 0 % | 0 % | 0 % | 0 % | 0 % |

Le point le plus important est l'évolution du diagnostic. Les premiers runs échouent pour des raisons numériques. Les runs suivants échouent pour des raisons d'exploration et de reward design. HER v2 ne résout pas encore la tâche, mais il déplace enfin le problème vers une ouverture partielle observable.
