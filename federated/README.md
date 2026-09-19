# federated/

Owns: simulated zone clients, the small event classifier, the Flower server,
FedAvg (and optional FedProx) aggregation.

Reads: `docs/schemas/event_feature_vector.json`-shaped records (per-zone local
data — real once `vision/` lands, synthetic before that so this track isn't
blocked).
Produces: a global classifier used at the incident-decision stage, plus round
metrics (round number, strategy, participating clients, accuracy, comms cost).

## Planned layout

```
federated/
  client.py        # Flower client: local train on one zone's features
  server.py         # Flower server: FedAvg / FedProx aggregation
  model.py          # small MLP / logistic classifier
  simulate_clients.py  # generates 3-5 simulated zone datasets
```
