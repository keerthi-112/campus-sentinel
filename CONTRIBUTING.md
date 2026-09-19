# Working on this repo

One shared repo, four tracks. Nobody commits straight to `main`.

## Per feature

```
git checkout main
git pull
git checkout -b <track>/<short-description>
# ...work, commit as you go...
git push origin <track>/<short-description>
```

Then open a Pull Request into `main` on GitHub. One teammate reviews and merges.
Afterwards, everyone else does `git checkout main && git pull` to pick it up.

## Branch prefixes

- `vision/...` — video, YOLO, zone/crowd logic
- `fl/...` — federated clients, classifier, Flower server
- `rag/...` — knowledge base, embeddings, retrieval, LLM prompts
- `app/...` — backend API, frontend, Docker Compose

## Keeping merges painless

Each track mostly lives in its own top-level folder, so conflicts should be rare.
The one shared surface is `docs/schemas/` — if you need to change a field there,
say so before merging, since it affects every other track.

Push and commit often on your own branch (no restriction there); keep PRs into
`main` small and frequent rather than one huge merge per phase.
