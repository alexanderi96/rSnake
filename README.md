# snake-rs

Neuroevolution sandbox: a population of snakes controlled by small feed-forward networks, evolved with **MAP-Elites** on procedurally generated terrain, rendered live in Bevy with an inspector for looking inside any agent's brain while it plays.

No gradient descent, no ML framework — the networks are flat `f32` genomes evaluated on the CPU, and the whole population runs in parallel via rayon.

## Why MAP-Elites

MAP-Elites is a quality-diversity algorithm: instead of converging on one best solution, it keeps a grid of elites, one per behavioral niche. Two snakes with the same score but different playstyles both survive. The result is an archive you can browse — "the best snake that hugs walls", "the best snake that spins in place" — rather than a single champion whose strategy you can't inspect.

The archive is a 33³ grid over three behavioral descriptors:

| Descriptor | Meaning |
|---|---|
| `turn_rate` | how often the snake changes direction |
| `center_affinity` | how much it sticks to the center of the map vs the edges |
| `coverage` | how much of the reachable grid it visits |

Each cell keeps the highest-fitness individual that lands in it. New individuals come from mutation and crossover of archive elites.

## The agent

Fully connected network, no bias toward any architecture search:

```
34 inputs → 128 → 64 → 3 outputs      (12931 genome parameters)
```

The 34 inputs are two stacked frames of a 17-value sensor vector — 8 obstacle rays, 8 target directions, 1 distance. Frame stacking gives the network a sense of motion that a single snapshot can't provide. The 3 outputs are relative moves: left, straight, right.

Genome color is inherited from the parent, so lineages are visually traceable across the population — you can watch a successful family spread through the grid.

## Terrain

Walls are generated with thresholded value-noise fBm plus a few cellular-automata smoothing passes — organic clusters, not mazes or rooms. `terrain_fill_rate` controls density, `terrain_blob_scale` cluster size, with a guaranteed clear radius around the spawn.

## Modes

- **Generational**: the whole population is evaluated, then the archive is updated and a new batch is sampled.
- **Continuous** (`--continuous`): a dead snake is immediately replaced by a fresh offspring on the same seed, so the world never resets and learning is uninterrupted.

Runs live under `runs/<uuid>/`, with the archive auto-saved as gzipped JSON on an interval. `--new-run` forces a fresh one instead of resuming.

## Inspector

Press `I` while running to open the brain inspector: sensor values, per-layer activations, and the decision of the selected agent, updated every step. `L` toggles the leaderboard, `K` the keybindings panel, arrows cycle through agents, and `3`–`9` jump directly to one. The archive itself can be browsed as a 3D point cloud, each point an elite positioned by its descriptors.

## Structure

```
src/main.rs                  CLI, run lifecycle, evaluation loop
src/config.rs                Hyperparameters (TOML + CLI overrides)
src/snake.rs                 sensors, fitness, save/load
src/plugins/simulation/      grid, step logic, snake entities
src/plugins/map_elites/      archive, individual (network + genome), evolution
src/plugins/terrain/         fBm cluster noise generation
src/plugins/brain_inspector/ inspector UI, gizmos, 3D archive view
src/plugins/ui/              HUD, panels, stats
src/plugins/food_spawn/      food placement
```

## Stack

| Component | Tech |
|---|---|
| Language | Rust 2021 |
| Engine / rendering | Bevy 0.13 |
| Parallel evaluation | rayon |
| Neural network | hand-rolled, no ML crate |
| Serialization | serde + serde_json + flate2 (gzip) |
| Config | toml + clap |
| RNG | rand (`SmallRng`, thread-local) |
| Profiling (optional) | pprof (flamegraph), dhat (heap), tracy |

## Running

```bash
cargo run --release                          # resume the latest run
cargo run --release -- --new-run             # start fresh
cargo run --release -- --continuous          # continuous replacement mode
cargo run --release -- -c config.toml        # load hyperparameters
```

Every hyperparameter in the config file can be overridden on the command line (`--population-size`, `--mutation-rate`, `--terrain-fill-rate`, …).

See [PROFILING.md](PROFILING.md) for the profiling workflow.

> `config.example.toml` still lists DQN parameters (`learning_rate`, `gamma`, `batch_size`) from an earlier gradient-based version. They're ignored — the live parameters are the ones in `src/config.rs`.
