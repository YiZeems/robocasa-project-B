# Analyse des courbes et des runs SAC

Ce document regroupe l'analyse des courbes d'entraînement les plus importantes. Il complète le rapport de projet avec une lecture plus expérimentale : que montre chaque métrique, pourquoi certains runs ont été arrêtés, et quel changement technique a été tenté ensuite.

Les figures sources sont dans [`docs/courbes/`](courbes/). Le fichier [`docs/courbes/readme_combined.md`](courbes/readme_combined.md) conserve aussi une version centrée sur les graphes combinés.

## Runs comparés

| Couleur | Run | Changement principal | Steps analysés |
|---|---|---|---:|
| Rouge | SAC v1 | `ent_coef="auto"` | 400k sur 500k |
| Orange | SAC v2 | `ent_coef="auto_0.1"`, `target_entropy=-4` | 400k sur 900k |
| Bleu | SAC v3 | `ent_coef=0.1` fixe, SDE | 400k |
| Vert | SAC v3 curriculum | seuil de succès réduit, spawn moins variable | 400k sur 500k |
| Turquoise | SAC HER v2 | HER + reward dense hybride | 300k |

HER v1 n'est pas central dans les figures combinées : en reward sparse pur, les courbes restent quasiment plates et ne changent pas le diagnostic visuel.

## Angle maximal de porte

![val_door_angle_max_mean](courbes/combined/combined_val_door_angle_max.png)

`val_door_angle_max_mean` mesure le meilleur angle atteint pendant un épisode de validation. C'est la métrique la plus parlante pour cette tâche, car elle décrit directement l'effet physique de la politique.

Les runs v1, v2 et v3 restent sous 0.02 rad. La porte ne bouge presque pas. Le curriculum monte plus haut, avec un pic autour de 0.039 rad, mais ce signal ne devient pas stable. HER v2 est le premier run qui dépasse franchement le bruit expérimental : 0.133 rad à 200k steps.

Ce point est important : même si le succès strict reste à 0 %, HER v2 montre une interaction réelle avec la porte. Les runs précédents échouaient avant même d'obtenir ce contact utile.

## Entropie SAC

![train_ent_coef](courbes/combined/combined_train_ent_coef.png)

Le coefficient d'entropie `alpha` contrôle l'exploration dans SAC. Les runs v1 et v2 montrent un effondrement clair : `alpha` descend vers zéro. La politique devient alors trop déterministe trop tôt, ce qui bloque l'exploration et rend les Q-values instables.

Le passage à `ent_coef=0.1` fixe en v3 est la première correction importante. À partir de ce moment, l'entraînement devient beaucoup plus sain : l'entropie reste stable, les critics ne divergent plus de façon catastrophique, et l'actor loss cesse de remonter vers le positif.

## Critic loss

![train_critic_loss](courbes/combined/combined_train_critic_loss.png)

La critic loss indique si les Q-values apprises sont crédibles. Sur v1 et v2, elle atteint des valeurs pathologiques, jusqu'à des dizaines de milliers. Dans ces conditions, l'actor ne reçoit plus un signal d'apprentissage exploitable.

Sur v3 et les runs suivants, la critic loss reste beaucoup plus basse. Cela ne veut pas dire que la tâche est résolue, mais cela déplace le diagnostic : le problème n'est plus principalement numérique, il devient comportemental. L'agent reste stable, mais il ne découvre pas assez souvent l'action de contact utile.

## Return de validation

![val_return_mean](courbes/combined/combined_val_return_mean.png)

Le return doit être lu avec prudence, car les rewards changent entre les configurations. HER v2, par exemple, utilise une composante sparse négative qui place son return sur une autre échelle.

La tendance reste informative :

- v1 et v2 stagnent avec des critics instables.
- v3 est stable mais ne progresse presque pas.
- Le curriculum produit un pic transitoire.
- HER v2 améliore progressivement son return, même sans atteindre le succès strict.

Le return seul ne suffit donc pas à juger un run. Il doit toujours être comparé à l'angle de porte et aux métriques anti-hacking.

## Hover-hacking et reward d'approche

![val_approach_frac_mean](courbes/combined/combined_val_approach_frac.png)

`val_approach_frac_mean` mesure la part du reward venant de l'approche de la poignée. Une valeur élevée indique que l'agent optimise surtout le fait de rester près de la poignée, sans forcément ouvrir.

Ce phénomène apparaît dans plusieurs runs. HER v2 est intéressant parce qu'il commence avec une `approach_frac` faible au meilleur checkpoint, puis cette métrique remonte à 300k steps. Cette remontée correspond à la baisse de `door_angle_max`, ce qui donne un diagnostic cohérent : après avoir appris à bouger la porte, la politique dérive vers un minimum local plus simple, rester proche de la poignée.

La suite logique est de réduire ou supprimer `w_approach` dans la configuration HER suivante, et de renforcer les rewards liées au progrès réel et au maintien de l'ouverture.

## Actor loss

![train_actor_loss](courbes/combined/combined_train_actor_loss.png)

Dans SAC, une actor loss négative et stable est généralement acceptable. Une remontée forte, surtout vers des valeurs positives, indique que l'actor apprend à suivre de mauvaises estimations de Q-value.

Le contraste est net :

| Run | Lecture de l'actor loss |
|---|---|
| v1 | descend puis remonte fortement, entraînement cassé |
| v2 | même trajectoire, divergence retardée |
| v3 | négative et stable, entraînement sain |
| curriculum | stable mais sans succès |
| HER v2 | amélioration jusqu'à 200k, puis légère remontée liée au hover-hacking |

Cette métrique a servi de critère d'arrêt : quand l'actor loss remonte en positif, prolonger le run n'apporte généralement rien.

## Chronologie expérimentale

```text
SAC v1
  Problème : ent_coef auto -> alpha vers 0 -> critic loss énorme
  Résultat : aucun apprentissage utile

SAC v2
  Changement : auto_0.1 et target_entropy modifiée
  Résultat : même crash, seulement retardé

SAC v3
  Changement : ent_coef fixe à 0.1
  Résultat : stabilité retrouvée, mais cold start

SAC v3 curriculum
  Changement : seuil plus facile, spawn réduit
  Résultat : léger pic, pas de convergence

SAC HER v1
  Changement : HER sparse pur
  Résultat : pas assez de guidage vers la poignée

SAC HER v2
  Changement : HER + reward dense hybride
  Résultat : première ouverture visible, puis hover-hacking émergent
```

## Tableau de synthèse

| Métrique | Objectif | v1 | v2 | v3 | Curriculum | HER v2 @200k |
|---|---:|---:|---:|---:|---:|---:|
| `door_angle_max` | > 0.10 rad | 0.014 | 0.017 | 0.012 | 0.039 | **0.133** |
| `ent_coef` | stable | crash | crash | 0.1 | 0.1 | 0.1 |
| `critic_loss` | contrôlée | 48k+ | 116k+ | ~35 | ~10 | ~5 |
| `val_return_mean` | croissant | stagne | stagne | stagne | pic ponctuel | **croissant** |
| `approach_frac` | faible | élevé | élevé | moyen | moyen | **faible** |
| `val_success_rate` | > 0 % | 0 % | 0 % | 0 % | 0 % | 0 % |

La conclusion expérimentale est donc nuancée. SAC n'a pas encore résolu la tâche au sens strict, mais les courbes montrent une progression réelle du diagnostic : d'abord corriger l'instabilité de SAC, ensuite traiter le cold start, enfin réduire le reward hacking dans HER.
