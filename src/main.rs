use bevy::prelude::*;
use rand::prelude::*;

// --- Costanti di gioco ---
const ARENA_WIDTH: i32 = 20;
const ARENA_HEIGHT: i32 = 20;
const SNAKE_START_LEN: usize = 3;
const SNAKE_STEP_SECS: f32 = 0.12;

// --- Componenti ---
#[derive(Component, Copy, Clone, Eq, PartialEq, Debug)]
struct Position { x: i32, y: i32 }

#[derive(Component, Copy, Clone)]
struct Size { w: f32, h: f32 }

#[derive(Component)]
struct SnakeHead;

#[derive(Component)]
struct SnakeSegment;

#[derive(Component)]
struct Food;

// --- Risorse & Eventi ---
#[derive(Resource, Default)]
struct SnakeSegments(Vec<Entity>);

#[derive(Resource, Default, Copy, Clone)]
struct SnakeDirection(Vec2I);

#[derive(Default, Copy, Clone)]
struct Vec2I { x: i32, y: i32 }

impl Vec2I {
    const UP: Self = Self { x: 0, y: 1 };
    const DOWN: Self = Self { x: 0, y: -1 };
    const LEFT: Self = Self { x: -1, y: 0 };
    const RIGHT: Self = Self { x: 1, y: 0 };
}

#[derive(Resource)]
struct MoveTimer(Timer);

#[derive(Event)]
struct AteFood;

#[derive(Event)]
struct GameOver;

// --- Plugin principale ---
fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Snake (Bevy)".into(),
                resolution: (600.0, 600.0).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.06, 0.07, 0.08))) // sfondo
        .insert_resource(SnakeSegments::default())
        .insert_resource(SnakeDirection(Vec2I::RIGHT))
        .insert_resource(MoveTimer(Timer::from_seconds(SNAKE_STEP_SECS, TimerMode::Repeating)))
        .add_event::<AteFood>()
        .add_event::<GameOver>()
        // Setup
        .add_systems(Startup, setup)
        .add_systems(Startup, spawn_snake)
        .add_systems(Startup, spawn_food)
        // Input ogni frame
        .add_systems(Update, keyboard_input)
        // Logica a passo fisso
        .add_systems(FixedUpdate, (
            tick_move_timer,
            snake_movement.run_if(should_step),
            eat_food.after(snake_movement),
            grow_snake.after(eat_food),
            check_collisions.after(snake_movement),
            respawn_food.run_if(resource_changed::<FoodSpawnNeeded>),
        ))
        .init_resource::<FoodSpawnNeeded>()
        .run();
}

// --- Setup camera ---
fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
}

// --- Converte coordinate griglia -> mondo ---
fn position_to_translation(pos: Position, win: &Window) -> Vec3 {
    let tile_w = win.width() / ARENA_WIDTH as f32;
    let tile_h = win.height() / ARENA_HEIGHT as f32;
    let x = (pos.x as f32 - ARENA_WIDTH as f32 / 2.0 + 0.5) * tile_w;
    let y = (pos.y as f32 - ARENA_HEIGHT as f32 / 2.0 + 0.5) * tile_h;
    Vec3::new(x, y, 0.0)
}

fn size_to_scale(size: Size, win: &Window) -> Vec3 {
    Vec3::new(
        size.w / ARENA_WIDTH as f32 * win.width(),
        size.h / ARENA_HEIGHT as f32 * win.height(),
        1.0,
    )
}

// --- Spawn snake iniziale ---
fn spawn_snake(
    mut commands: Commands,
    mut segments_res: ResMut<SnakeSegments>,
    windows: Query<&Window>,
) {
    let win = windows.single();
    let head_pos = Position { x: ARENA_WIDTH / 2, y: ARENA_HEIGHT / 2 };

    // head
    let head = commands
        .spawn((
            SpriteBundle {
                sprite: Sprite {
                    color: Color::srgb_u8(40, 230, 140),
                    ..default()
                },
                transform: Transform {
                    translation: position_to_translation(head_pos, win),
                    scale: size_to_scale(Size { w: 0.95, h: 0.95 }, win),
                    ..default()
                },
                ..default()
            },
            SnakeHead,
            SnakeSegment,
            head_pos,
            Size { w: 0.95, h: 0.95 },
        ))
        .id();

    let mut last = head_pos;
    let mut entities = vec![head];

    for i in 1..SNAKE_START_LEN {
        let p = Position { x: last.x - 1, y: last.y };
        let seg = commands
            .spawn((
                SpriteBundle {
                    sprite: Sprite {
                        color: Color::srgb_u8(20, 200, 110),
                        ..default()
                    },
                    transform: Transform {
                        translation: position_to_translation(p, win),
                        scale: size_to_scale(Size { w: 0.9, h: 0.9 }, win),
                        ..default()
                    },
                    ..default()
                },
                SnakeSegment,
                p,
                Size { w: 0.9, h: 0.9 },
            ))
            .id();

        entities.push(seg);
        last = p;
        let _ = i; // silence unused for clarity
    }

    segments_res.0 = entities;
}

// --- Input direzione ---
fn keyboard_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut dir: ResMut<SnakeDirection>,
) {
    let d = dir.0;
    if keys.just_pressed(KeyCode::ArrowUp) || keys.just_pressed(KeyCode::KeyW) {
        if d != Vec2I::DOWN { dir.0 = Vec2I::UP; }
    } else if keys.just_pressed(KeyCode::ArrowDown) || keys.just_pressed(KeyCode::KeyS) {
        if d != Vec2I::UP { dir.0 = Vec2I::DOWN; }
    } else if keys.just_pressed(KeyCode::ArrowLeft) || keys.just_pressed(KeyCode::KeyA) {
        if d != Vec2I::RIGHT { dir.0 = Vec2I::LEFT; }
    } else if keys.just_pressed(KeyCode::ArrowRight) || keys.just_pressed(KeyCode::KeyD) {
        if d != Vec2I::LEFT { dir.0 = Vec2I::RIGHT; }
    }
}

// --- Timer a cadenza fissa (FixedUpdate) ---
fn tick_move_timer(
    time: Res<Time>,
    mut timer: ResMut<MoveTimer>,
) {
    timer.0.tick(time.delta());
}

// Condizione: esegui passo quando scatta il timer
fn should_step(timer: Res<MoveTimer>) -> bool {
    timer.0.finished_this_tick()
}

// --- Movimento Snake ---
fn snake_movement(
    dir: Res<SnakeDirection>,
    mut query: Query<(&mut Position, &mut Transform), With<SnakeSegment>>,
    windows: Query<&Window>,
    mut segments: ResMut<SnakeSegments>,
) {
    if query.is_empty() { return; }

    let win = windows.single();

    // Snapshot posizioni correnti
    let mut positions: Vec<Position> = segments.0
        .iter()
        .map(|e| query.get_mut(*e).ok().map(|(p, _)| *p).unwrap())
        .collect();

    // nuova testa
    let mut new_head = positions[0];
    new_head.x += dir.0.x;
    new_head.y += dir.0.y;

    // wrap ai bordi
    if new_head.x < 0 { new_head.x = ARENA_WIDTH - 1; }
    if new_head.x >= ARENA_WIDTH { new_head.x = 0; }
    if new_head.y < 0 { new_head.y = ARENA_HEIGHT - 1; }
    if new_head.y >= ARENA_HEIGHT { new_head.y = 0; }

    // shift corpo
    for i in (1..positions.len()).rev() {
        positions[i] = positions[i - 1];
    }
    positions[0] = new_head;

    // applica posizioni e trasform
    for (i, e) in segments.0.clone().iter().enumerate() {
        if let Ok((mut p, mut t)) = query.get_mut(*e) {
            *p = positions[i];
            t.translation = position_to_translation(*p, win);
        }
    }
}

// --- Cibo ---
#[derive(Resource, Default)]
struct FoodSpawnNeeded(bool);

fn spawn_food(mut commands: Commands, windows: Query<&Window>) {
    let win = windows.single();
    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::srgb_u8(230, 70, 70),
                ..default()
            },
            transform: Transform {
                // placeholder, verrà posizionato da respawn_food
                translation: Vec3::ZERO,
                scale: size_to_scale(Size { w: 0.8, h: 0.8 }, win),
                ..default()
            },
            ..default()
        },
        Food,
        Position { x: 0, y: 0 },
        Size { w: 0.8, h: 0.8 },
    ));
}

fn respawn_food(
    mut food_q: Query<(&mut Position, &mut Transform), With<Food>>,
    windows: Query<&Window>,
    mut need: ResMut<FoodSpawnNeeded>,
    segments: Res<SnakeSegments>,
    seg_q: Query<&Position, With<SnakeSegment>>,
) {
    if !need.0 { return; }
    need.0 = false;

    let win = windows.single();
    let occupied: std::collections::HashSet<(i32, i32)> = segments.0.iter()
        .filter_map(|e| seg_q.get(*e).ok())
        .map(|p| (p.x, p.y))
        .collect();

    let mut rng = thread_rng();
    // trova una cella libera
    let (mut fx, mut fy);
    loop {
        fx = rng.gen_range(0..ARENA_WIDTH);
        fy = rng.gen_range(0..ARENA_HEIGHT);
        if !occupied.contains(&(fx, fy)) { break; }
    }

    if let Ok((mut p, mut t)) = food_q.get_single_mut() {
        *p = Position { x: fx, y: fy };
        t.translation = position_to_translation(*p, win);
    }
}

// --- Mangia & Cresci ---
fn eat_food(
    mut commands: Commands,
    mut need: ResMut<FoodSpawnNeeded>,
    mut ate_writer: EventWriter<AteFood>,
    head_q: Query<&Position, (With<SnakeHead>, With<SnakeSegment>)>,
    food_q: Query<&Position, With<Food>>,
) {
    if let (Ok(hp), Ok(fp)) = (head_q.get_single(), food_q.get_single()) {
        if hp.x == fp.x && hp.y == fp.y {
            ate_writer.send(AteFood);
            need.0 = true;
        }
    }
}

fn grow_snake(
    mut reader: EventReader<AteFood>,
    mut commands: Commands,
    windows: Query<&Window>,
    segments: Res<SnakeSegments>,
    mut segments_res: ResMut<SnakeSegments>,
    query: Query<(&Position, &Size), With<SnakeSegment>>,
) {
    if reader.read().next().is_none() { return; }

    let win = windows.single();

    // trova ultima posizione del corpo per aggiungere un nuovo segmento in coda
    if let Some(&last_entity) = segments.0.last() {
        if let Ok((last_pos, last_size)) = query.get(last_entity) {
            let new_entity = commands
                .spawn((
                    SpriteBundle {
                        sprite: Sprite {
                            color: Color::srgb_u8(20, 200, 110),
                            ..default()
                        },
                        transform: Transform {
                            translation: position_to_translation(*last_pos, win),
                            scale: size_to_scale(*last_size, win),
                            ..default()
                        },
                        ..default()
                    },
                    SnakeSegment,
                    *last_pos,
                    *last_size,
                ))
                .id();
            let mut v = segments.0.clone();
            v.push(new_entity);
            segments_res.0 = v;
        }
    }
}

// --- Collisioni con se stessi (Game Over) ---
fn check_collisions(
    mut game_over: EventWriter<GameOver>,
    segments: Res<SnakeSegments>,
    q: Query<&Position, With<SnakeSegment>>,
) {
    if segments.0.len() < 4 { return; }
    let head = segments.0[0];
    let head_pos = q.get(head).ok().copied();
    if head_pos.is_none() { return; }
    let head_pos = head_pos.unwrap();

    for &e in segments.0.iter().skip(1) {
        if let Ok(p) = q.get(e) {
            if p.x == head_pos.x && p.y == head_pos.y {
                game_over.send(GameOver);
                // semplice reset: tronca a 3 segmenti (opzionale potresti ricaricare la scena)
                // Qui potresti implementare una logica di reset completa; per semplicità non facciamo nulla.
                break;
            }
        }
    }
}
