//! Brain Loading Module
//!
//! Handles loading trained brains from json.gz files and injecting them as Bevy Resources.
//! Supports both archive files and individual brain files.

use bevy::prelude::*;
use std::path::Path;

use crate::plugins::map_elites::archive::MapElitesArchive;
use crate::plugins::map_elites::individual::{Brain, Individual, GENOME_SIZE};
use crate::snake::load_json_gz;

// ============================================================================
// RESOURCES
// ============================================================================

/// Resource holding a loaded brain for inspection
#[allow(dead_code)]
#[derive(Resource, Clone)]
pub struct LoadedBrain {
    /// The brain data
    pub brain: Brain,
    /// Optional metadata about the brain
    pub metadata: BrainMetadata,
}

/// Metadata for a loaded brain
#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
pub struct BrainMetadata {
    /// Source file path
    pub source_path: String,
    /// Fitness score if known
    pub fitness: Option<f32>,
    /// Turn rate (descriptor 1) if known
    pub desc_turn_rate: Option<f32>,
    /// Center affinity (descriptor 2) if known
    pub desc_center_affinity: Option<f32>,
    /// Coverage (descriptor 3) if known
    pub desc_coverage: Option<f32>,
    /// Generation if from archive
    pub generation: Option<u32>,
    /// Cell coordinates in archive (x, y, z)
    pub archive_cell: Option<(usize, usize, usize)>,
}

/// Resource holding a loaded archive
#[allow(dead_code)]
#[derive(Resource)]
pub struct LoadedArchive {
    /// The loaded archive
    pub archive: MapElitesArchive,
    /// Source file path
    pub source_path: String,
}

/// State for brain loading operations
#[derive(Resource, Default)]
pub struct BrainLoaderState {
    /// Last error message if loading failed
    pub last_error: Option<String>,
    /// Last loaded file path
    pub last_loaded_path: Option<String>,
    /// Whether a file is currently being loaded
    pub is_loading: bool,
}

// ============================================================================
// LOADING FUNCTIONS
// ============================================================================

/// Load a brain from a json.gz file
///
/// Supports:
/// - Individual brain JSON files
/// - MAP-Elites archive files (returns best brain)
/// - Training session files
///
/// # Errors
/// Returns an error if the file cannot be read or parsed.
pub fn load_brain_from_gz(path: &Path) -> Result<(Brain, BrainMetadata), BrainLoadError> {
    // Try to load as an archive first
    if let Ok(archive) = MapElitesArchive::load(path.to_str().unwrap_or("")) {
        // Find the best brain in the archive
        if let Some((cell, individual)) = archive
            .grid
            .iter()
            .max_by_key(|(_, ind)| ind.fitness as i32)
        {
            let metadata = BrainMetadata {
                source_path: path.to_string_lossy().to_string(),
                fitness: Some(individual.fitness),
                desc_turn_rate: Some(individual.desc_turn_rate),
                desc_center_affinity: Some(individual.desc_center_affinity),
                desc_coverage: Some(individual.desc_coverage),
                generation: Some(archive.generation),
                archive_cell: Some(*cell),
            };
            return Ok((individual.brain.clone(), metadata));
        }
        return Err(BrainLoadError::EmptyArchive);
    }

    // Try to load as an Individual
    if let Ok(individual) = load_json_gz::<Individual>(path) {
        // Verify genome size
        if individual.brain.genome.len() != GENOME_SIZE {
            return Err(BrainLoadError::GenomeSizeMismatch {
                expected: GENOME_SIZE,
                found: individual.brain.genome.len(),
            });
        }

        let metadata = BrainMetadata {
            source_path: path.to_string_lossy().to_string(),
            fitness: Some(individual.fitness),
            desc_turn_rate: Some(individual.desc_turn_rate),
            desc_center_affinity: Some(individual.desc_center_affinity),
            desc_coverage: Some(individual.desc_coverage),
            generation: None,
            archive_cell: None,
        };
        return Ok((individual.brain, metadata));
    }

    // Try to load as a raw brain (just the genome)
    if let Ok(brain) = load_json_gz::<Brain>(path) {
        if brain.genome.len() != GENOME_SIZE {
            return Err(BrainLoadError::GenomeSizeMismatch {
                expected: GENOME_SIZE,
                found: brain.genome.len(),
            });
        }

        let metadata = BrainMetadata {
            source_path: path.to_string_lossy().to_string(),
            ..Default::default()
        };
        return Ok((brain, metadata));
    }

    Err(BrainLoadError::UnknownFormat)
}

/// Load an entire archive from a json.gz file
///
/// # Errors
/// Returns an error if the file cannot be read or parsed as an archive.
pub fn load_archive_from_gz(path: &Path) -> Result<MapElitesArchive, BrainLoadError> {
    MapElitesArchive::load(path.to_str().unwrap_or(""))
        .map_err(|e| BrainLoadError::IoError(e.to_string()))
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

/// Errors that can occur during brain loading
#[derive(Debug, Clone)]
pub enum BrainLoadError {
    /// File I/O error
    IoError(String),
    /// Invalid file format
    UnknownFormat,
    /// Empty archive (no individuals)
    EmptyArchive,
    /// Genome size doesn't match current architecture
    GenomeSizeMismatch { expected: usize, found: usize },
    /// JSON parsing error
    ParseError(String),
}

impl std::fmt::Display for BrainLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BrainLoadError::IoError(msg) => write!(f, "IO error: {}", msg),
            BrainLoadError::UnknownFormat => write!(f, "Unknown file format"),
            BrainLoadError::EmptyArchive => write!(f, "Archive is empty"),
            BrainLoadError::GenomeSizeMismatch { expected, found } => {
                write!(
                    f,
                    "Genome size mismatch: expected {}, found {}",
                    expected, found
                )
            }
            BrainLoadError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for BrainLoadError {}

impl From<std::io::Error> for BrainLoadError {
    fn from(e: std::io::Error) -> Self {
        BrainLoadError::IoError(e.to_string())
    }
}

impl From<serde_json::Error> for BrainLoadError {
    fn from(e: serde_json::Error) -> Self {
        BrainLoadError::ParseError(e.to_string())
    }
}

// ============================================================================
// BEVY SYSTEMS
// ============================================================================

/// System to load a brain from a file path
///
/// This can be triggered by a UI button or keyboard shortcut.
pub fn load_brain_system(
    mut commands: Commands,
    mut loader_state: ResMut<BrainLoaderState>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    run_dir: Res<crate::snake::RunDirectory>,
) {
    // Load brain on Ctrl+L
    if keyboard_input.pressed(KeyCode::ControlLeft) && keyboard_input.just_pressed(KeyCode::KeyL) {
        // Default to loading the archive from the current run directory
        let archive_path = run_dir.0.join("archive.json.gz");

        if !archive_path.exists() {
            loader_state.last_error =
                Some(format!("Archive not found at {}", archive_path.display()));
            return;
        }

        loader_state.is_loading = true;
        loader_state.last_error = None;

        match load_brain_from_gz(&archive_path) {
            Ok((brain, metadata)) => {
                println!(
                    "✅ Loaded brain from {} (fitness: {:.1})",
                    metadata.source_path,
                    metadata.fitness.unwrap_or(0.0)
                );

                commands.insert_resource(LoadedBrain { brain, metadata });
                loader_state.last_loaded_path = Some(archive_path.to_string_lossy().to_string());
                loader_state.last_error = None;
            }
            Err(e) => {
                eprintln!("❌ Failed to load brain: {}", e);
                loader_state.last_error = Some(e.to_string());
            }
        }

        loader_state.is_loading = false;
    }
}

/// System to load the full archive
pub fn load_archive_system(
    mut commands: Commands,
    mut loader_state: ResMut<BrainLoaderState>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    run_dir: Res<crate::snake::RunDirectory>,
) {
    // Load full archive on Ctrl+Shift+L
    if keyboard_input.pressed(KeyCode::ControlLeft)
        && keyboard_input.pressed(KeyCode::ShiftLeft)
        && keyboard_input.just_pressed(KeyCode::KeyL)
    {
        let archive_path = run_dir.0.join("archive.json.gz");

        if !archive_path.exists() {
            loader_state.last_error =
                Some(format!("Archive not found at {}", archive_path.display()));
            return;
        }

        loader_state.is_loading = true;
        loader_state.last_error = None;

        match load_archive_from_gz(&archive_path) {
            Ok(archive) => {
                println!(
                    "✅ Loaded archive with {} individuals",
                    archive.filled_cells()
                );

                commands.insert_resource(LoadedArchive {
                    archive,
                    source_path: archive_path.to_string_lossy().to_string(),
                });
                loader_state.last_loaded_path = Some(archive_path.to_string_lossy().to_string());
                loader_state.last_error = None;
            }
            Err(e) => {
                eprintln!("❌ Failed to load archive: {}", e);
                loader_state.last_error = Some(e.to_string());
            }
        }

        loader_state.is_loading = false;
    }
}

/// Initialize brain loader resources
pub fn setup_brain_loader(mut commands: Commands) {
    commands.insert_resource(BrainLoaderState::default());
}

// ============================================================================
// PLUGIN
// ============================================================================

/// Plugin for brain loading functionality
pub struct BrainLoaderPlugin;

impl Plugin for BrainLoaderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_brain_loader)
            .add_systems(Update, (load_brain_system, load_archive_system));
    }
}
