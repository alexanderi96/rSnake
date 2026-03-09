//! 3D MAP-Elites Archive Cube Visualization
//!
//! Renders the 3-dimensional behavioural archive as an interactive voxel cube
//! using a dedicated render-target + camera isolated on its own RenderLayer.
//!
//! Features
//! ─────────
//! • Each filled cell → a semi-transparent PBR cube.
//!   Alpha  = 0.04 + (fitness / max_fitness) * 0.88  (nearly invisible → opaque)
//!   Colour = rgba(0.1, t, 1-t, alpha)  – same green→blue ramp as the 2-D view.
//! • Mouse drag in the inspector panel orbits the scene.
//! • On mouse-release a "magnet" snaps to the nearest canonical view
//!   (6 face-aligned + 4 isometric corners) if the angle is < ~23°.
//! • Axis markers + bounding-box edges are rendered inside the 3-D scene,
//!   so they always rotate with the cube.
//!
//! Integration
//! ───────────
//! 1. Add `pub mod archive3d;` to `brain_inspector/mod.rs`.
//! 2. Add `Archive3dPlugin` to your app builder.
//! 3. In `brain_inspector/ui.rs` → `update_inspector_content` system:
//!      - Add `archive3d_target: Res<Archive3dTarget>` parameter.
//!      - Pass `&archive3d_target` to `spawn_map_elites_tab`.
//! 4. Replace `spawn_map_elites_tab` body with the version at the bottom
//!    of this file (search "// ── UI INTEGRATION ──").

use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
    render::{
        camera::RenderTarget,
        render_resource::{
            Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
        },
        view::RenderLayers,
    },
};
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use crate::brain_inspector::{BrainInspectorState, InspectorTab};
use crate::evolution::EvolutionManager;
use crate::ui::PanelVisibility;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Isolated render layer – nothing else renders here.
pub const LAYER: u8 = 3;

/// Dimensions of the off-screen render target (pixels).
pub const RT_W: u32 = 430;
pub const RT_H: u32 = 390;

/// Visual half-size of each cell cube.
pub const CELL_SIZE: f32 = 0.80;

/// Grid step (centre-to-centre distance between adjacent cells).
pub const CELL_STEP: f32 = 1.05;

/// Perspective FOV (radians). Narrower → less perspective distortion.
pub const FOV: f32 = 0.50;

/// Camera distance from origin.
pub const CAM_DIST: f32 = 28.0;

/// Camera min/max distance for zoom limits.
pub const CAM_DIST_MIN: f32 = 15.0;
pub const CAM_DIST_MAX: f32 = 200.0;

/// Mouse sensitivity (rad / pixel).
pub const DRAG_SENS: f32 = 0.006;

/// Mouse wheel zoom sensitivity.
pub const ZOOM_SENS: f32 = 0.15;

/// cos(angle) threshold to trigger magnetic snap. cos(23°) ≈ 0.921.
pub const SNAP_COS: f32 = 0.921;

/// Slerp speed for snap animation (s⁻¹).
pub const SNAP_SPEED: f32 = 9.0;

/// Slerp speed while freely rotating (s⁻¹).
pub const FREE_SPEED: f32 = 22.0;

// ============================================================================
// RESOURCES
// ============================================================================

/// GPU texture that the archive camera renders into.
/// Store the handle in the UI's `ImageBundle` to display the result.
#[derive(Resource)]
pub struct Archive3dTarget {
    pub image: Handle<Image>,
}

/// View mode for the 3D archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewMode {
    /// Free rotation in all directions.
    FreeRotate,
    /// Locked perpendicular to a specific face (X+, X-, Y+, Y-, Z+, Z-).
    LockedToFace,
}

/// Face indices for locked mode (corresponds to canonical views).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Face {
    XPos, // +X face
    XNeg, // -X face
    YPos, // +Y face (top)
    YNeg, // -Y face (bottom)
    ZPos, // +Z face
    ZNeg, // -Z face
}

impl Face {
    /// Get the quaternion that looks at this face from outside the cube.
    pub fn to_quaternion(self) -> Quat {
        match self {
            Face::XPos => Quat::from_rotation_y(-FRAC_PI_2),
            Face::XNeg => Quat::from_rotation_y(FRAC_PI_2),
            Face::YPos => Quat::from_rotation_x(FRAC_PI_2),
            Face::YNeg => Quat::from_rotation_x(-FRAC_PI_2),
            Face::ZPos => Quat::IDENTITY,
            Face::ZNeg => Quat::from_rotation_y(PI),
        }
    }

    /// Get the display name for this face.
    pub fn name(&self) -> &'static str {
        match self {
            Face::XPos => "X+ (Turn+)",
            Face::XNeg => "X- (Turn-)",
            Face::YPos => "Y+ (Expl+)",
            Face::YNeg => "Y- (Expl-)",
            Face::ZPos => "Z+ (ObsHug+)",
            Face::ZNeg => "Z- (ObsHug-)",
        }
    }

    /// Get the next face when pressing "next" in locked mode.
    pub fn next(&self) -> Self {
        match self {
            Face::XPos => Face::YPos,
            Face::YPos => Face::ZPos,
            Face::ZPos => Face::XNeg,
            Face::XNeg => Face::YNeg,
            Face::YNeg => Face::ZNeg,
            Face::ZNeg => Face::XPos,
        }
    }

    /// Get the previous face when pressing "previous" in locked mode.
    pub fn previous(&self) -> Self {
        match self {
            Face::XPos => Face::ZNeg,
            Face::ZPos => Face::YPos,
            Face::YPos => Face::XPos,
            Face::XNeg => Face::ZPos,
            Face::ZNeg => Face::YNeg,
            Face::YNeg => Face::XNeg,
        }
    }
}

/// Orbit state for the archive cube (shared between input and render systems).
#[derive(Resource)]
pub struct Archive3dOrbit {
    /// Smoothed current rotation applied to the scene root.
    pub rotation: Quat,
    /// Desired orientation – may be ahead of `rotation` during snap.
    pub target: Quat,
    /// True while the left mouse button is held inside the panel.
    pub dragging: bool,
    /// True while animating toward a snap orientation.
    pub snapping: bool,
    /// Camera distance (for zoom).
    pub camera_distance: f32,
    /// Current view mode.
    pub view_mode: ViewMode,
    /// Current face when locked.
    pub locked_face: Face,
}

impl Default for Archive3dOrbit {
    fn default() -> Self {
        // Pleasant initial view: front-right-top isometric.
        let q = Quat::from_euler(EulerRot::YXZ, FRAC_PI_4 * 0.85, -0.55, 0.0);
        Self {
            rotation: q,
            target: q,
            dragging: false,
            snapping: false,
            camera_distance: CAM_DIST,
            view_mode: ViewMode::FreeRotate,
            locked_face: Face::ZPos,
        }
    }
}

// ============================================================================
// COMPONENTS
// ============================================================================

#[derive(Component)]
pub struct Archive3dCam;
#[derive(Component)]
pub struct Archive3dRoot;
#[derive(Component)]
pub struct Archive3dCell;
#[derive(Component)]
pub struct Archive3dDecor; // axes + bbox edges

// ============================================================================
// SETUP
// ============================================================================

pub fn setup_archive3d(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    // ── render-target texture ──────────────────────────────────────────────
    let size = Extent3d {
        width: RT_W,
        height: RT_H,
        depth_or_array_layers: 1,
    };
    let mut img = Image {
        texture_descriptor: TextureDescriptor {
            label: Some("archive3d_rt"),
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };
    img.resize(size);
    let rt = images.add(img);

    // ── camera ────────────────────────────────────────────────────────────
    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                target: RenderTarget::Image(rt.clone()),
                order: -1,
                ..default()
            },
            camera_3d: Camera3d::default(),
            transform: Transform::from_xyz(0.0, 0.0, CAM_DIST).looking_at(Vec3::ZERO, Vec3::Y),
            projection: Projection::Perspective(PerspectiveProjection {
                fov: FOV,
                ..default()
            }),
            ..default()
        },
        RenderLayers::layer(LAYER),
        Archive3dCam,
    ));

    // ── key light ─────────────────────────────────────────────────────────
    commands.spawn((
        DirectionalLightBundle {
            directional_light: DirectionalLight {
                color: Color::WHITE,
                illuminance: 18_000.0,
                shadows_enabled: false,
                ..default()
            },
            transform: Transform::from_xyz(3.0, 5.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        RenderLayers::layer(LAYER),
    ));

    // ── fill light (opposite side, cooler tint) ───────────────────────────
    commands.spawn((
        DirectionalLightBundle {
            directional_light: DirectionalLight {
                color: Color::rgb(0.65, 0.75, 1.0),
                illuminance: 6_000.0,
                shadows_enabled: false,
                ..default()
            },
            transform: Transform::from_xyz(-3.0, -2.0, -4.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        RenderLayers::layer(LAYER),
    ));

    // ── scene rotation root ───────────────────────────────────────────────
    commands.spawn((
        SpatialBundle::default(),
        RenderLayers::layer(LAYER),
        Archive3dRoot,
    ));

    commands.insert_resource(Archive3dTarget { image: rt });
    commands.insert_resource(Archive3dOrbit::default());
}

// ============================================================================
// CUBE REBUILD
// ============================================================================

/// Clears and re-spawns archive cells whenever the archive changes.
/// Also spawns axis arrows and bounding-box edges as part of the 3-D scene.
pub fn rebuild_archive_cubes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    evo_manager: Res<EvolutionManager>,
    old_cells: Query<Entity, Or<(With<Archive3dCell>, With<Archive3dDecor>)>>,
    root_q: Query<Entity, With<Archive3dRoot>>,
) {
    if !evo_manager.is_changed() {
        return;
    }

    for e in old_cells.iter() {
        commands.entity(e).despawn_recursive();
    }

    let Ok(root) = root_q.get_single() else {
        return;
    };

    let archive = &evo_manager.archive;
    let res = crate::map_elites::GRID_RESOLUTION;
    let max_fit = archive.best_fitness.max(1.0);
    let half = (res as f32 - 1.0) * CELL_STEP * 0.5;

    // Shared mesh for all cells
    #[allow(deprecated)]
    let cell_mesh: Handle<Mesh> = meshes.add(shape::Cube { size: CELL_SIZE });

    let mut children: Vec<Entity> = Vec::with_capacity(archive.filled_cells() + 50);

    // ── archive cells ──────────────────────────────────────────────────────
    for (&(x, y, z), ind) in archive.grid.iter() {
        let t = (ind.fitness / max_fit).clamp(0.0, 1.0);

        // Alpha: low-fitness cells are ghostly; high-fitness cells are solid.
        let alpha = 0.04 + t * 0.88;

        // Same hue ramp as the existing 2-D view: (0.1, t, 1-t)
        let mat: Handle<StandardMaterial> = materials.add(StandardMaterial {
            base_color: Color::rgba(0.1, t, 1.0 - t, alpha),
            alpha_mode: AlphaMode::Blend,
            // Faint emissive so near-zero cells are still barely perceptible.
            emissive: Color::rgba(0.01, t * 0.12, (1.0 - t) * 0.08, 0.0),
            perceptual_roughness: 0.55,
            metallic: 0.05,
            double_sided: true,
            cull_mode: None,
            ..default()
        });

        let pos = Vec3::new(
            x as f32 * CELL_STEP - half,
            y as f32 * CELL_STEP - half,
            z as f32 * CELL_STEP - half,
        );

        children.push(
            commands
                .spawn((
                    PbrBundle {
                        mesh: cell_mesh.clone(),
                        material: mat,
                        transform: Transform::from_translation(pos),
                        ..default()
                    },
                    RenderLayers::layer(LAYER),
                    Archive3dCell,
                ))
                .id(),
        );
    }

    // ── bounding-box edges (12 thin boxes) ────────────────────────────────
    let edge_mat: Handle<StandardMaterial> = materials.add(StandardMaterial {
        base_color: Color::rgba(0.45, 0.50, 0.75, 0.28),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        double_sided: true,
        cull_mode: None,
        ..default()
    });

    let bbox_half = half + CELL_STEP * 0.5;
    let edge_len = bbox_half * 2.0;
    let edge_thick = 0.07;

    // 4 edges along X
    #[allow(deprecated)]
    let x_mesh: Handle<Mesh> = meshes.add(shape::Box::new(edge_len, edge_thick, edge_thick));
    for &(sy, sz) in &[(-1.0_f32, -1.0_f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        children.push(spawn_edge(
            &mut commands,
            x_mesh.clone(),
            edge_mat.clone(),
            Vec3::new(0.0, sy * bbox_half, sz * bbox_half),
        ));
    }
    // 4 edges along Y
    #[allow(deprecated)]
    let y_mesh: Handle<Mesh> = meshes.add(shape::Box::new(edge_thick, edge_len, edge_thick));
    for &(sx, sz) in &[(-1.0_f32, -1.0_f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        children.push(spawn_edge(
            &mut commands,
            y_mesh.clone(),
            edge_mat.clone(),
            Vec3::new(sx * bbox_half, 0.0, sz * bbox_half),
        ));
    }
    // 4 edges along Z
    #[allow(deprecated)]
    let z_mesh: Handle<Mesh> = meshes.add(shape::Box::new(edge_thick, edge_thick, edge_len));
    for &(sx, sy) in &[(-1.0_f32, -1.0_f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        children.push(spawn_edge(
            &mut commands,
            z_mesh.clone(),
            edge_mat.clone(),
            Vec3::new(sx * bbox_half, sy * bbox_half, 0.0),
        ));
    }

    // ── axis arrows (thin colored elongated boxes) ─────────────────────────
    // Origin: front-bottom-left corner of the bounding box
    let ax_origin = Vec3::new(-bbox_half, -bbox_half, -bbox_half);
    let ax_len = edge_len * 0.55;
    let ax_thick = 0.18;

    // X = Turn Rate (red)
    children.push(spawn_axis_arrow(
        &mut commands,
        &mut meshes,
        &mut materials,
        ax_origin + Vec3::X * ax_len * 0.5,
        Vec3::new(ax_len, ax_thick, ax_thick),
        Color::rgba(1.0, 0.25, 0.25, 0.90),
    ));
    // Y = Body Pressure (green)
    children.push(spawn_axis_arrow(
        &mut commands,
        &mut meshes,
        &mut materials,
        ax_origin + Vec3::Y * ax_len * 0.5,
        Vec3::new(ax_thick, ax_len, ax_thick),
        Color::rgba(0.25, 1.0, 0.25, 0.90),
    ));
    // Z = Turn Alternation (blue)
    children.push(spawn_axis_arrow(
        &mut commands,
        &mut meshes,
        &mut materials,
        ax_origin + Vec3::Z * ax_len * 0.5,
        Vec3::new(ax_thick, ax_thick, ax_len),
        Color::rgba(0.30, 0.55, 1.0, 0.90),
    ));

    commands.entity(root).push_children(&children);
}

fn spawn_edge(
    commands: &mut Commands,
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    pos: Vec3,
) -> Entity {
    commands
        .spawn((
            PbrBundle {
                mesh,
                material,
                transform: Transform::from_translation(pos),
                ..default()
            },
            RenderLayers::layer(LAYER),
            Archive3dDecor,
        ))
        .id()
}

fn spawn_axis_arrow(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    pos: Vec3,
    size: Vec3,
    color: Color,
) -> Entity {
    #[allow(deprecated)]
    let mesh: Handle<Mesh> = meshes.add(shape::Box::new(size.x, size.y, size.z));
    let mat: Handle<StandardMaterial> = materials.add(StandardMaterial {
        base_color: color,
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
    commands
        .spawn((
            PbrBundle {
                mesh,
                material: mat,
                transform: Transform::from_translation(pos),
                ..default()
            },
            RenderLayers::layer(LAYER),
            Archive3dDecor,
        ))
        .id()
}

// ============================================================================
// ORBIT / MOUSE / KEYBOARD INTERACTION
// ============================================================================

/// Processes mouse drag, wheel zoom, and keyboard input for the archive cube.
pub fn update_archive3d_orbit(
    mut orbit: ResMut<Archive3dOrbit>,
    mut root_q: Query<&mut Transform, With<Archive3dRoot>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut mouse_wheel: EventReader<MouseWheel>,
    windows: Query<&Window>,
    panel_visibility: Res<PanelVisibility>,
    inspector_state: Res<BrainInspectorState>,
    time: Res<Time>,
) {
    // Always drain the event queues.
    let mut delta = Vec2::ZERO;
    for ev in mouse_motion.read() {
        delta += ev.delta;
    }

    let mut wheel_delta = 0.0;
    for ev in mouse_wheel.read() {
        wheel_delta += ev.y;
    }

    if !panel_visibility.inspector || inspector_state.active_tab != InspectorTab::MapElites {
        return;
    }

    let Ok(window) = windows.get_single() else {
        return;
    };

    // Inspector panel: right edge at window.width()-10, width = 480px.
    let panel_x_min = window.width() - 10.0 - 480.0;
    let cursor_in_panel = window
        .cursor_position()
        .map(|p| p.x >= panel_x_min)
        .unwrap_or(false);

    // ── keyboard input for mode toggle and face navigation ──
    // Toggle mode: 'V' = Toggle view mode (free/locked)
    if keyboard_input.just_pressed(KeyCode::KeyV) {
        match orbit.view_mode {
            ViewMode::FreeRotate => {
                orbit.view_mode = ViewMode::LockedToFace;
                orbit.target = orbit.locked_face.to_quaternion();
                orbit.rotation = orbit.target;
                orbit.snapping = false;
                println!(
                    "[ARCHIVE3D] Mode: Locked to face ({})",
                    orbit.locked_face.name()
                );
            }
            ViewMode::LockedToFace => {
                orbit.view_mode = ViewMode::FreeRotate;
                println!("[ARCHIVE3D] Mode: Free Rotate");
            }
        }
    }

    // Face navigation (only in locked mode): [ and ] keys
    if orbit.view_mode == ViewMode::LockedToFace {
        if keyboard_input.just_pressed(KeyCode::BracketRight) {
            orbit.locked_face = orbit.locked_face.next();
            orbit.target = orbit.locked_face.to_quaternion();
            orbit.snapping = true;
            println!("[ARCHIVE3D] Face: {}", orbit.locked_face.name());
        }
        if keyboard_input.just_pressed(KeyCode::BracketLeft) {
            orbit.locked_face = orbit.locked_face.previous();
            orbit.target = orbit.locked_face.to_quaternion();
            orbit.snapping = true;
            println!("[ARCHIVE3D] Face: {}", orbit.locked_face.name());
        }
    }

    // ── mouse wheel zoom (applied via scale of root) ──
    let current_scale = orbit.camera_distance / 28.0;
    let target_scale = if wheel_delta != 0.0 && cursor_in_panel {
        (orbit.camera_distance - wheel_delta * ZOOM_SENS * orbit.camera_distance)
            .clamp(CAM_DIST_MIN, CAM_DIST_MAX)
            / 28.0
    } else {
        current_scale
    };
    orbit.camera_distance = (target_scale * 28.0).clamp(CAM_DIST_MIN, CAM_DIST_MAX);

    // ── drag start / end ──
    if mouse_buttons.just_pressed(MouseButton::Left) && cursor_in_panel {
        orbit.dragging = true;
        orbit.snapping = false;
    }
    if mouse_buttons.just_released(MouseButton::Left) && orbit.dragging {
        orbit.dragging = false;
        // Only attempt snap in free-rotate mode
        if orbit.view_mode == ViewMode::FreeRotate {
            attempt_snap(&mut orbit);
        }
    }

    // ── apply drag rotation ──
    if orbit.dragging && delta.length_squared() > 0.0 {
        if orbit.view_mode == ViewMode::FreeRotate {
            // Full free rotation
            let yaw = Quat::from_rotation_y(-delta.x * DRAG_SENS);
            let pitch = Quat::from_rotation_x(-delta.y * DRAG_SENS);
            orbit.target = (yaw * orbit.target * pitch).normalize();
            orbit.snapping = false;
        } else {
            // In locked mode: rotate around Y axis (viewer's vertical)
            // This orbits around the cube from current viewing angle
            let yaw = Quat::from_rotation_y(-delta.x * DRAG_SENS * 2.0);
            orbit.target = (yaw * orbit.target).normalize();
            orbit.snapping = false;
        }
    }

    // ── smooth slerp towards target ──
    let speed = if orbit.snapping {
        SNAP_SPEED
    } else {
        FREE_SPEED
    };
    let alpha = (speed * time.delta_seconds()).min(1.0);
    orbit.rotation = orbit.rotation.slerp(orbit.target, alpha);

    // Terminate snap when converged.
    if orbit.snapping && orbit.rotation.dot(orbit.target).abs() > 0.9999 {
        orbit.rotation = orbit.target;
        orbit.snapping = false;
    }

    // ── write to root transform (rotation + scale for zoom) ──
    for mut t in root_q.iter_mut() {
        t.rotation = orbit.rotation;
        t.scale = Vec3::splat(target_scale);
    }
}

/// Checks whether the post-drag orientation is within `SNAP_COS` of any
/// canonical viewpoint; if so, animates toward it.
fn attempt_snap(orbit: &mut Archive3dOrbit) {
    let canonical: [Quat; 10] = [
        // 6 face-perpendicular views
        Quat::IDENTITY,
        Quat::from_rotation_y(PI),
        Quat::from_rotation_y(FRAC_PI_2),
        Quat::from_rotation_y(-FRAC_PI_2),
        Quat::from_rotation_x(-FRAC_PI_2),
        Quat::from_rotation_x(FRAC_PI_2),
        // 4 isometric corners
        Quat::from_euler(EulerRot::YXZ, FRAC_PI_4, -0.6155, 0.0),
        Quat::from_euler(EulerRot::YXZ, -FRAC_PI_4, -0.6155, 0.0),
        Quat::from_euler(EulerRot::YXZ, PI + FRAC_PI_4, -0.6155, 0.0),
        Quat::from_euler(EulerRot::YXZ, PI - FRAC_PI_4, -0.6155, 0.0),
    ];

    let cur = orbit.target;
    let best = canonical
        .iter()
        .copied()
        .max_by(|a, b| {
            cur.dot(*a)
                .abs()
                .partial_cmp(&cur.dot(*b).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    if cur.dot(best).abs() >= SNAP_COS {
        orbit.target = best;
        orbit.snapping = true;
    }
}

// ============================================================================
// ── UI INTEGRATION ──────────────────────────────────────────────────────────
// Replace the old `spawn_map_elites_tab` in brain_inspector/ui.rs with this.
// Also add `archive3d_target: Res<crate::brain_inspector::archive3d::Archive3dTarget>`
// to the `update_inspector_content` system parameters, and forward it below.
// ============================================================================

/// Replacement for `spawn_map_elites_tab` in `brain_inspector/ui.rs`.
pub fn spawn_map_elites_tab(
    parent: &mut ChildBuilder,
    evo_manager: &crate::evolution::EvolutionManager,
    archive3d_target: &Archive3dTarget,
    orbit: &Archive3dOrbit,
) {
    let archive = &evo_manager.archive;

    // ── stats header ──────────────────────────────────────────────────────
    parent.spawn(TextBundle::from_section(
        format!(
            "Gen {}  │  Coverage {:.1}%  ({}/{})",
            archive.generation,
            archive.coverage() * 100.0,
            archive.filled_cells(),
            archive.capacity(),
        ),
        TextStyle {
            font_size: 13.0,
            color: Color::WHITE,
            ..default()
        },
    ));
    parent.spawn(TextBundle::from_section(
        format!("Best fitness: {:.0}", archive.best_fitness),
        TextStyle {
            font_size: 12.0,
            color: Color::GOLD,
            ..default()
        },
    ));

    // ── axis legend ───────────────────────────────────────────────────────
    parent
        .spawn(NodeBundle {
            style: Style {
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(12.0),
                margin: UiRect::vertical(Val::Px(4.0)),
                ..default()
            },
            ..default()
        })
        .with_children(|row| {
            for (color, label) in [
                (Color::rgb(1.0, 0.25, 0.25), "X = Turn Rate"),
                (Color::rgb(0.25, 1.0, 0.25), "Y = Body Pressure"),
                (Color::rgb(0.30, 0.55, 1.0), "Z = Turn Alternation"),
            ] {
                row.spawn(NodeBundle {
                    style: Style {
                        width: Val::Px(10.0),
                        height: Val::Px(10.0),
                        margin: UiRect::right(Val::Px(3.0)),
                        align_self: AlignSelf::Center,
                        ..default()
                    },
                    background_color: color.into(),
                    ..default()
                });
                row.spawn(TextBundle::from_section(
                    label,
                    TextStyle {
                        font_size: 10.0,
                        color: Color::GRAY,
                        ..default()
                    },
                ));
            }
        });

    // ── mode indicator ────────────────────────────────────────────────────────
    let mode_text = match orbit.view_mode {
        ViewMode::FreeRotate => "[V] Toggle View  |  Drag: rotate  |  Scroll: zoom".to_string(),
        ViewMode::LockedToFace => format!(
            "[V] Toggle  |  Drag: orbit  |  [ ]: {}  |  Scroll: zoom",
            orbit.locked_face.name()
        ),
    };
    parent.spawn(TextBundle::from_section(
        mode_text,
        TextStyle {
            font_size: 10.0,
            color: Color::rgba(0.6, 0.6, 0.6, 0.8),
            ..default()
        },
    ));

    // ── 3-D render-target image ───────────────────────────────────────────
    parent.spawn(ImageBundle {
        style: Style {
            width: Val::Px(RT_W as f32),
            height: Val::Px(RT_H as f32),
            margin: UiRect::top(Val::Px(4.0)),
            ..default()
        },
        image: UiImage::new(archive3d_target.image.clone()),
        ..default()
    });

    // ── hint text ─────────────────────────────────────────────────────────
    let hint_text = match orbit.view_mode {
        ViewMode::FreeRotate => "Release near face/corner to snap",
        ViewMode::LockedToFace => "Drag rotates around face axis",
    };
    parent.spawn(TextBundle::from_section(
        hint_text,
        TextStyle {
            font_size: 10.0,
            color: Color::rgba(0.6, 0.6, 0.6, 0.8),
            ..default()
        },
    ));

    // ── colour scale legend ───────────────────────────────────────────────
    parent
        .spawn(NodeBundle {
            style: Style {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                column_gap: Val::Px(6.0),
                margin: UiRect::top(Val::Px(6.0)),
                ..default()
            },
            ..default()
        })
        .with_children(|row| {
            row.spawn(TextBundle::from_section(
                "low fitness",
                TextStyle {
                    font_size: 9.0,
                    color: Color::GRAY,
                    ..default()
                },
            ));
            // Gradient strip (approximated with discrete steps)
            for i in 0..20u8 {
                let t = i as f32 / 19.0;
                let alpha = 0.04 + t * 0.88;
                row.spawn(NodeBundle {
                    style: Style {
                        width: Val::Px(10.0),
                        height: Val::Px(10.0),
                        ..default()
                    },
                    background_color: Color::rgba(0.1, t, 1.0 - t, alpha).into(),
                    ..default()
                });
            }
            row.spawn(TextBundle::from_section(
                "high fitness",
                TextStyle {
                    font_size: 9.0,
                    color: Color::WHITE,
                    ..default()
                },
            ));
        });
}

// ============================================================================
// PLUGIN
// ============================================================================

pub struct Archive3dPlugin;

impl Plugin for Archive3dPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_archive3d)
            .add_systems(Update, (rebuild_archive_cubes, update_archive3d_orbit));
    }
}
