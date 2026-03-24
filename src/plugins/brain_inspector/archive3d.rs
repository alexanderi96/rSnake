//! 3D MAP-Elites Archive Cube Visualization
//!
//! Renders the 3-dimensional behavioural archive as an interactive voxel cube
//! using a dedicated render-target + camera isolated on its own RenderLayer.
//!
//! Rotation model: trackball — both axes in world space.
//!   horizontal drag → always Ry (world Y)
//!   vertical   drag → always Rx (world X = camera right, fixed)
//!   formula: new = Rx(pitch) * Ry(yaw) * old
//!
//! Axis-lock gizmo rings (Blender-style):
//!   Hold X → lock rotation to world X axis
//!   Hold Y → lock rotation to world Y axis
//!   Hold Z → lock rotation to world Z axis
//!   Release → back to free trackball

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
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

use super::{BrainInspectorState, InspectorTab};
use crate::plugins::map_elites::evolution::EvolutionManager;
use crate::plugins::ui::PanelVisibility;

// ============================================================================
// CONSTANTS
// ============================================================================

pub const LAYER: u8 = 3;
pub const RT_W: u32 = 430;
pub const RT_H: u32 = 390;
pub const CELL_SIZE: f32 = 0.80;
pub const CELL_STEP: f32 = 1.05;
pub const FOV: f32 = 0.50;
pub const CAM_DIST: f32 = 28.0;
pub const CAM_DIST_MIN: f32 = 15.0;
pub const CAM_DIST_MAX: f32 = 200.0;

/// Mouse sensitivity (rad / pixel).
pub const DRAG_SENS: f32 = 0.006;
/// Scroll zoom sensitivity.
pub const ZOOM_SENS: f32 = 0.15;
/// Maximum elevation angle (just under 90°).
const MAX_ELEV: f32 = FRAC_PI_2 * 0.98;

// ============================================================================
// DRAG AXIS
// ============================================================================

#[derive(Default, PartialEq, Clone, Copy, Debug)]
pub enum DragAxis {
    #[default]
    Free,
    X,
    Y,
    Z,
}

// ============================================================================
// RESOURCES
// ============================================================================

#[derive(Resource)]
pub struct Archive3dTarget {
    pub image: Handle<Image>,
}

/// Orbit state.
#[derive(Resource)]
pub struct Archive3dOrbit {
    /// Authoritative scene rotation.
    pub rotation: Quat,
    /// Approximate elevation for clamping (radians, negative = looking down).
    pub elevation: f32,
    pub dragging: bool,
    pub camera_distance: f32,
    /// Current drag axis constraint.
    pub drag_axis: DragAxis,
}

impl Default for Archive3dOrbit {
    fn default() -> Self {
        let az: f32 = FRAC_PI_4 * 0.85;
        let el: f32 = -0.55;
        let rotation = Quat::from_rotation_x(el) * Quat::from_rotation_y(az);
        Self {
            rotation,
            elevation: el,
            dragging: false,
            camera_distance: CAM_DIST,
            drag_axis: DragAxis::Free,
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
pub struct Archive3dDecor;

/// Marker for the three axis-lock ring gizmos.
#[derive(Component)]
pub struct AxisRing(pub DragAxis);

// ============================================================================
// SETUP
// ============================================================================

pub fn setup_archive3d(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // ── Render target ──────────────────────────────────────────────────────
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

    // ── Camera ─────────────────────────────────────────────────────────────
    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                target: RenderTarget::Image(rt.clone()),
                order: -1,
                ..default()
            },
            camera_3d: Camera3d::default(),
            transform: Transform::from_xyz(0.0, 0.0, CAM_DIST)
                .looking_at(Vec3::ZERO, Vec3::Y),
            projection: Projection::Perspective(PerspectiveProjection {
                fov: FOV,
                ..default()
            }),
            ..default()
        },
        RenderLayers::layer(LAYER),
        Archive3dCam,
    ));

    // ── Lights ─────────────────────────────────────────────────────────────
    commands.spawn((
        DirectionalLightBundle {
            directional_light: DirectionalLight {
                color: Color::WHITE,
                illuminance: 18_000.0,
                shadows_enabled: false,
                ..default()
            },
            transform: Transform::from_xyz(3.0, 5.0, 4.0)
                .looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        RenderLayers::layer(LAYER),
    ));

    commands.spawn((
        DirectionalLightBundle {
            directional_light: DirectionalLight {
                color: Color::rgb(0.65, 0.75, 1.0),
                illuminance: 6_000.0,
                shadows_enabled: false,
                ..default()
            },
            transform: Transform::from_xyz(-3.0, -2.0, -4.0)
                .looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        RenderLayers::layer(LAYER),
    ));

    // ── Scene rotation root ────────────────────────────────────────────────
    commands.spawn((
        SpatialBundle::default(),
        RenderLayers::layer(LAYER),
        Archive3dRoot,
    ));

    // ── Axis-lock gizmo rings ──────────────────────────────────────────────
    // Radius slightly outside the bounding box so rings are always visible.
    let res = crate::plugins::map_elites::GRID_RESOLUTION as f32;
    let ring_radius = res * CELL_STEP * 0.5 + 2.0;
    let ring_thickness = 0.14;

    #[allow(deprecated)]
    let ring_mesh = meshes.add(shape::Torus {
        radius: ring_radius,
        ring_radius: ring_thickness,
        subdivisions_segments: 64,
        subdivisions_sides: 12,
    });

    // X ring — red, lives in the YZ plane → rotate 90° around Z
    commands.spawn((
        PbrBundle {
            mesh: ring_mesh.clone(),
            material: materials.add(StandardMaterial {
                base_color: Color::rgba(1.0, 0.18, 0.18, 0.80),
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                double_sided: true,
                cull_mode: None,
                ..default()
            }),
            transform: Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            ..default()
        },
        RenderLayers::layer(LAYER),
        Archive3dDecor,
        AxisRing(DragAxis::X),
    ));

    // Y ring — green, lives in the XZ plane (identity)
    commands.spawn((
        PbrBundle {
            mesh: ring_mesh.clone(),
            material: materials.add(StandardMaterial {
                base_color: Color::rgba(0.18, 1.0, 0.18, 0.80),
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                double_sided: true,
                cull_mode: None,
                ..default()
            }),
            transform: Transform::IDENTITY,
            ..default()
        },
        RenderLayers::layer(LAYER),
        Archive3dDecor,
        AxisRing(DragAxis::Y),
    ));

    // Z ring — blue, lives in the XY plane → rotate 90° around X
    commands.spawn((
        PbrBundle {
            mesh: ring_mesh.clone(),
            material: materials.add(StandardMaterial {
                base_color: Color::rgba(0.18, 0.45, 1.0, 0.80),
                alpha_mode: AlphaMode::Blend,
                unlit: true,
                double_sided: true,
                cull_mode: None,
                ..default()
            }),
            transform: Transform::from_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            ..default()
        },
        RenderLayers::layer(LAYER),
        Archive3dDecor,
        AxisRing(DragAxis::Z),
    ));

    commands.insert_resource(Archive3dTarget { image: rt });
    commands.insert_resource(Archive3dOrbit::default());
}

// ============================================================================
// CUBE REBUILD
// ============================================================================

pub fn rebuild_archive_cubes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    evo_manager: Res<EvolutionManager>,
    old_cells: Query<Entity, With<Archive3dCell>>,
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
    let res = crate::plugins::map_elites::GRID_RESOLUTION;
    let max_fit = archive.best_fitness.max(1.0);
    let half = (res as f32 - 1.0) * CELL_STEP * 0.5;

    #[allow(deprecated)]
    let cell_mesh: Handle<Mesh> = meshes.add(shape::Cube { size: CELL_SIZE });

    let mut children: Vec<Entity> = Vec::with_capacity(archive.filled_cells() + 50);

    // ── Archive cells ──────────────────────────────────────────────────────
    for (&(x, y, z), ind) in archive.grid.iter() {
        let t = (ind.fitness / max_fit).clamp(0.0, 1.0);
        let alpha = 0.04 + t * 0.88;

        let mat: Handle<StandardMaterial> = materials.add(StandardMaterial {
            base_color: Color::rgba(0.1, t, 1.0 - t, alpha),
            alpha_mode: AlphaMode::Blend,
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

    // ── Bounding-box edges ─────────────────────────────────────────────────
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

    // ── Axis arrows ────────────────────────────────────────────────────────
    let ax_origin = Vec3::new(-bbox_half, -bbox_half, -bbox_half);
    let ax_len = edge_len * 0.55;
    let ax_thick = 0.18;

    children.push(spawn_axis_arrow(
        &mut commands,
        &mut meshes,
        &mut materials,
        ax_origin + Vec3::X * ax_len * 0.5,
        Vec3::new(ax_len, ax_thick, ax_thick),
        Color::rgba(1.0, 0.25, 0.25, 0.90),
    ));
    children.push(spawn_axis_arrow(
        &mut commands,
        &mut meshes,
        &mut materials,
        ax_origin + Vec3::Y * ax_len * 0.5,
        Vec3::new(ax_thick, ax_len, ax_thick),
        Color::rgba(0.25, 1.0, 0.25, 0.90),
    ));
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
// AXIS-LOCK HIGHLIGHT
// ============================================================================

/// Brighten the active ring, dim the others.
pub fn update_ring_highlight(
    orbit: Res<Archive3dOrbit>,
    mut ring_q: Query<(&AxisRing, &mut Handle<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !orbit.is_changed() {
        return;
    }

    for (ring, mat_handle) in ring_q.iter_mut() {
        let is_active = ring.0 == orbit.drag_axis;
        if let Some(mat) = materials.get_mut(&*mat_handle) {
            let (base, alpha) = match ring.0 {
                DragAxis::X => (Color::rgb(1.0, 0.18, 0.18), if is_active { 1.0 } else { 0.55 }),
                DragAxis::Y => (Color::rgb(0.18, 1.0, 0.18), if is_active { 1.0 } else { 0.55 }),
                DragAxis::Z => (Color::rgb(0.18, 0.45, 1.0), if is_active { 1.0 } else { 0.55 }),
                DragAxis::Free => (Color::WHITE, 0.55),
            };
            mat.base_color = base.with_a(alpha);
            // Glow when active
            mat.emissive = if is_active {
                match ring.0 {
                    DragAxis::X => Color::rgba(0.6, 0.0, 0.0, 0.0),
                    DragAxis::Y => Color::rgba(0.0, 0.6, 0.0, 0.0),
                    DragAxis::Z => Color::rgba(0.0, 0.1, 0.6, 0.0),
                    DragAxis::Free => Color::BLACK,
                }
            } else {
                Color::BLACK
            };
        }
    }
}

// ============================================================================
// AXIS-LOCK HOTKEYS
// ============================================================================

/// X / Y / Z keys lock the drag axis while held.
/// Releasing the key (when not currently dragging) resets to Free.
pub fn set_drag_axis(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut orbit: ResMut<Archive3dOrbit>,
    panel_visibility: Res<PanelVisibility>,
    inspector_state: Res<BrainInspectorState>,
) {
    // Only active when the MAP-Elites tab is open
    if !panel_visibility.inspector || inspector_state.active_tab != InspectorTab::MapElites {
        return;
    }

    if keyboard.just_pressed(KeyCode::KeyX) {
        orbit.drag_axis = DragAxis::X;
    } else if keyboard.just_pressed(KeyCode::KeyY) {
        orbit.drag_axis = DragAxis::Y;
    } else if keyboard.just_pressed(KeyCode::KeyZ) {
        orbit.drag_axis = DragAxis::Z;
    }

    // Reset to free when key is released and mouse is not held
    let axis_key_released = keyboard.just_released(KeyCode::KeyX)
        || keyboard.just_released(KeyCode::KeyY)
        || keyboard.just_released(KeyCode::KeyZ);

    if axis_key_released && !orbit.dragging {
        orbit.drag_axis = DragAxis::Free;
    }
}

// ============================================================================
// ORBIT INPUT
// ============================================================================

pub fn update_archive3d_orbit(
    mut orbit: ResMut<Archive3dOrbit>,
    mut root_q: Query<&mut Transform, With<Archive3dRoot>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut mouse_wheel: EventReader<MouseWheel>,
    windows: Query<&Window>,
    panel_visibility: Res<PanelVisibility>,
    inspector_state: Res<BrainInspectorState>,
) {
    // Always drain events to avoid stale accumulation.
    let mut delta = Vec2::ZERO;
    for ev in mouse_motion.read() {
        delta += ev.delta;
    }
    let mut wheel_delta = 0.0_f32;
    for ev in mouse_wheel.read() {
        wheel_delta += ev.y;
    }

    if !panel_visibility.inspector || inspector_state.active_tab != InspectorTab::MapElites {
        return;
    }

    let Ok(window) = windows.get_single() else {
        return;
    };

    let panel_x_min = window.width() - 10.0 - 480.0;
    let cursor_in_panel = window
        .cursor_position()
        .map(|p| p.x >= panel_x_min)
        .unwrap_or(false);

    // ── Zoom ───────────────────────────────────────────────────────────────
    if wheel_delta != 0.0 && cursor_in_panel {
        orbit.camera_distance = (orbit.camera_distance
            - wheel_delta * ZOOM_SENS * orbit.camera_distance)
            .clamp(CAM_DIST_MIN, CAM_DIST_MAX);
    }

    // ── Drag start/end ─────────────────────────────────────────────────────
    if mouse_buttons.just_pressed(MouseButton::Left) && cursor_in_panel {
        orbit.dragging = true;
    }
    if mouse_buttons.just_released(MouseButton::Left) {
        orbit.dragging = false;
        // Reset axis lock when drag ends
        orbit.drag_axis = DragAxis::Free;
    }

    // ── Rotation ───────────────────────────────────────────────────────────
    if orbit.dragging && delta.length_squared() > 0.0 {
        match orbit.drag_axis {
            DragAxis::Free => {
                // World-space trackball: yaw (Y) + clamped pitch (X)
                let yaw_delta = delta.x * DRAG_SENS;
                let raw_el = orbit.elevation - delta.y * DRAG_SENS;
                let clamped_el = raw_el.clamp(-MAX_ELEV, MAX_ELEV);
                let pitch_delta = clamped_el - orbit.elevation;
                orbit.elevation = clamped_el;

                let yaw = Quat::from_rotation_y(yaw_delta);
                let pitch = Quat::from_rotation_x(pitch_delta);
                orbit.rotation = (pitch * yaw * orbit.rotation).normalize();
            }
            DragAxis::X => {
                // Horizontal drag → spin around world X
                let angle = delta.x * DRAG_SENS;
                orbit.rotation = (Quat::from_rotation_x(angle) * orbit.rotation).normalize();
            }
            DragAxis::Y => {
                // Horizontal drag → spin around world Y
                let angle = delta.x * DRAG_SENS;
                orbit.rotation = (Quat::from_rotation_y(angle) * orbit.rotation).normalize();
            }
            DragAxis::Z => {
                // Horizontal drag → spin around world Z
                let angle = delta.x * DRAG_SENS;
                orbit.rotation = (Quat::from_rotation_z(angle) * orbit.rotation).normalize();
            }
        }
    }

    // ── Write to root ──────────────────────────────────────────────────────
    let zoom_scale = orbit.camera_distance / CAM_DIST;
    for mut t in root_q.iter_mut() {
        t.rotation = orbit.rotation;
        t.scale = Vec3::splat(zoom_scale);
    }
}

// ============================================================================
// UI
// ============================================================================

pub fn spawn_map_elites_tab(
    parent: &mut ChildBuilder,
    evo_manager: &crate::plugins::map_elites::evolution::EvolutionManager,
    archive3d_target: &Archive3dTarget,
    orbit: &Archive3dOrbit,
) {
    let archive = &evo_manager.archive;

    // Stats header
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

    // Axis legend
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
                (Color::rgb(1.0, 0.25, 0.25), "X = Path Efficiency"),
                (Color::rgb(0.25, 1.0, 0.25), "Y = Danger Affinity"),
                (Color::rgb(0.30, 0.55, 1.0), "Z = Spatial Spread"),
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

    // Axis lock indicator + controls hint
    let axis_label = match orbit.drag_axis {
        DragAxis::Free => "Drag: free rotate  |  Scroll: zoom",
        DragAxis::X    => "Axis locked: X (red)  |  drag to rotate",
        DragAxis::Y    => "Axis locked: Y (green)  |  drag to rotate",
        DragAxis::Z    => "Axis locked: Z (blue)  |  drag to rotate",
    };
    let axis_color = match orbit.drag_axis {
        DragAxis::Free => Color::rgba(0.6, 0.6, 0.6, 0.8),
        DragAxis::X    => Color::rgba(1.0, 0.4, 0.4, 1.0),
        DragAxis::Y    => Color::rgba(0.4, 1.0, 0.4, 1.0),
        DragAxis::Z    => Color::rgba(0.4, 0.6, 1.0, 1.0),
    };
    parent.spawn(TextBundle::from_section(
        axis_label,
        TextStyle {
            font_size: 10.0,
            color: axis_color,
            ..default()
        },
    ));
    parent.spawn(TextBundle::from_section(
        "Hold [X] [Y] [Z] to lock rotation axis",
        TextStyle {
            font_size: 10.0,
            color: Color::rgba(0.5, 0.5, 0.5, 0.7),
            ..default()
        },
    ));

    // 3-D render-target image
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

    // Colour scale legend
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
            .add_systems(
                Update,
                (
                    rebuild_archive_cubes,
                    set_drag_axis,
                    update_archive3d_orbit.after(set_drag_axis),
                    update_ring_highlight.after(set_drag_axis),
                ),
            );
    }
}
