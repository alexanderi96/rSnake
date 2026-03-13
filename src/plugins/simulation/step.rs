//! Simulation step logic
//!
//! Contains the core simulation step functions used by the main game loop.

pub use super::snake::{
    bfs_distance, get_current_17_state, GameState, GenerationSeed, GridDimensions, GridMap,
    Position, SnakeInstance, BASE_STATE_SIZE,
};

use crate::config::Hyperparameters;
use crate::plugins::food_spawn::FoodSpawnZone;
use crate::plugins::map_elites::evolution::{EvolutionManager, GenerationRecord};
use crate::plugins::map_elites::individual::Action;
use crate::snake::GlobalTrainingHistory;

/// Apply moves to all snakes for one simulation step.
/// This is the core game logic that updates snake positions, handles collisions,
/// food eating, and game over conditions.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn apply_moves(
    snakes: &mut [SnakeInstance],
    grid_map: &mut GridMap,
    grid: &GridDimensions,
    current_moves: &mut [Option<(Action, [f32; BASE_STATE_SIZE])>],
    config: &Hyperparameters,
    gen_seed: &GenerationSeed,
    snake_vs_snake: bool,
    food_spawn_zone: &FoodSpawnZone,
) -> (u32, bool) {
    let mut new_high_score = 0u32;

    // Rebuild grid_map
    grid_map.clear();
    for (idx, snake) in snakes.iter().enumerate() {
        if !snake.is_game_over {
            let cell_val = ((idx + 1) as u16).min(255) as u8;
            for pos in snake.snake.iter() {
                grid_map.set(pos.x, pos.y, cell_val);
            }
        }
    }

    for (idx, result) in current_moves.iter().enumerate() {
        let Some((action, current_17)) = result else {
            continue;
        };
        let snake = &mut snakes[idx];
        if snake.is_game_over {
            continue;
        }

        snake.previous_state = *current_17;

        match action {
            Action::Left => {
                snake.direction = snake.direction.turn_left();
            }
            Action::Right => {
                snake.direction = snake.direction.turn_right();
            }
            Action::Straight => {}
        }

        let (dx, dy) = snake.direction.as_vec();
        let old_head = snake.snake[0];
        let new_head = Position {
            x: old_head.x + dx,
            y: old_head.y + dy,
        };

        let ate_food = new_head == snake.food;

        // Tail exception
        let tail_pos = snake.snake.back().copied();
        let is_self_collision =
            snake.body_set.contains(&new_head) && (ate_food || Some(new_head) != tail_pos);

        let is_collision = is_self_collision
            || if snake_vs_snake {
                grid_map.is_collision(new_head.x, new_head.y, snake.id)
            } else {
                grid_map.is_wall_collision(new_head.x, new_head.y)
            };

        // BFS-aware timeout: usa la distanza BFS reale invece della dimensione della mappa
        let is_timeout = snake.steps_without_food
            > config.calculate_timeout_bfs(snake.snake.len(), snake.food_real_distance);

        if !is_collision && !is_timeout {
            // Dense progress signal: calcola PRIMA di incrementare steps
            let progress_ratio = if snake.food_real_distance > 0 {
                let par = snake.food_real_distance as f32 * 2.0;
                ((par - snake.steps_without_food as f32) / par).clamp(0.0, 1.0)
            } else {
                1.0
            };
            snake.path_progress_sum += progress_ratio;

            // POI incrementa i contatori
            snake.steps_without_food += 1;
            snake.frames_survived += 1;

            snake.visited_cells.insert((new_head.x, new_head.y));

            // Calculate obstacle proximity using raycast sensors
            let obstacle_proximity = current_17[0..8].iter().sum::<f32>() / 8.0;
            snake.obstacle_adjacency_sum += obstacle_proximity;
        }

        if is_collision || is_timeout {
            snake.is_game_over = true;
        } else {
            snake.snake.push_front(new_head);
            snake.body_set.insert(new_head);
            if ate_food {
                // Calcola efficienza PRIMA di resettare steps_without_food
                if snake.food_real_distance > 0 && snake.steps_without_food > 0 {
                    let efficiency = (snake.food_real_distance as f32
                        / snake.steps_without_food as f32)
                        .clamp(0.0, 1.0);
                    snake.path_directness_sum += efficiency;
                }
                snake.score += 1;
                if snake.score > new_high_score {
                    new_high_score = snake.score;
                }

                // Calcola la coda (si libererà al prossimo step)
                let tail = snake.snake.back().copied();

                // Trova nuovo cibo
                let zone_center = Some((food_spawn_zone.center.x, food_spawn_zone.center.y));
                let new_food = gen_seed.food_at_free(
                    snake.score as usize,
                    &snake.body_set,
                    &grid_map.terrain,
                    grid.width,
                    zone_center,
                    food_spawn_zone.radius,
                );

                // Calcola distanza BFS reale
                let real_dist = bfs_distance(
                    new_head,
                    new_food,
                    &snake.body_set,
                    tail,
                    &grid_map.terrain,
                    grid.width,
                    grid.height,
                );

                match real_dist {
                    None => {
                        // Nessun path verso il cibo → kill immediato
                        snake.is_game_over = true;
                    }
                    Some(dist) => {
                        snake.food = new_food;
                        snake.food_real_distance = dist;
                        snake.steps_without_food = 0;
                    }
                }
            } else {
                let tail = *snake.snake.back().unwrap();
                snake.body_set.remove(&tail);
                snake.snake.pop_back();
            }
        }
    }

    (new_high_score, snakes.iter().all(|s| s.is_game_over))
}

/// End of generation handler - computes fitness, updates evolution, etc.
#[allow(dead_code)]
pub fn end_generation(
    game: &mut GameState,
    evo_manager: &mut EvolutionManager,
    global_history: &mut GlobalTrainingHistory,
    grid: &GridDimensions,
) -> GenerationRecord {
    for (i, snake) in game.snakes.iter().enumerate() {
        if let Some(ind) = evo_manager.get_individual_mut(i) {
            ind.fitness = snake.fitness(grid);
            ind.desc_path_efficiency = snake.path_efficiency();
            ind.desc_danger_affinity = snake.danger_affinity();
            ind.desc_spatial_spread = snake.spatial_spread();
            ind.frames_survived = snake.frames_survived;
            ind.apples_eaten = snake.score;
            ind.is_alive = false;
        }
    }

    let gen_high_score = game.snakes.iter().map(|s| s.score).max().unwrap_or(0);

    let mut record = evo_manager.end_generation();
    record.generation_high_score = gen_high_score;

    if gen_high_score > global_history.all_time_high_score {
        global_history.all_time_high_score = gen_high_score;
    }

    global_history.push(record.clone());

    // Limit current session records to prevent memory growth
    if global_history.current_session.len() > 10_000 {
        global_history.current_session.drain(0..5_000);
    }

    println!(
        "Gen {:4} | Fitness: {:.0} (best: {:.0}) | Coverage: {:.1}% | {:.2}s",
        record.generation,
        record.avg_fitness,
        record.best_fitness,
        record.archive_coverage * 100.0,
        record.elapsed_secs
    );

    evo_manager.start_generation();

    record
}

/// Compute moves for all snakes using parallel processing.
/// Returns a vector of (action, state) for each snake.
#[allow(dead_code)]
pub fn compute_moves(
    snakes: &[SnakeInstance],
    grid_map: &GridMap,
    grid: &GridDimensions,
    population: &[std::sync::Arc<crate::plugins::map_elites::individual::Brain>],
    snake_vs_snake: bool,
) -> Vec<Option<(Action, [f32; BASE_STATE_SIZE])>> {
    use rayon::prelude::*;
    let mut results = vec![None; snakes.len()];
    results
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, result)| {
            let snake = &snakes[idx];
            if snake.is_game_over {
                return;
            }
            let brain = match population.get(snake.id) {
                Some(b) => b.as_ref(),
                None => return,
            };
            let current_17 = get_current_17_state(snake, grid_map, grid, snake_vs_snake);
            let mut state_34 = [0.0f32; crate::plugins::simulation::STATE_SIZE];
            state_34[..17].copy_from_slice(&current_17);
            state_34[17..].copy_from_slice(&snake.previous_state);
            *result = Some((brain.predict(&state_34), current_17));
        });
    results
}
