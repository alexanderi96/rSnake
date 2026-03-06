//! Procedural Terrain Generation — fBm cluster noise
//!
//! Unico algoritmo: value noise fBm sogliato + CA di rifinitura bordi.
//! Niente maze, rooms, barriers: solo cluster organici.

use crate::snake::Position;

pub struct TerrainMap {
    pub width: i32,
    pub height: i32,
    pub cells: Vec<bool>,
}

impl TerrainMap {
    pub fn new(width: i32, height: i32) -> Self {
        Self {
            width,
            height,
            cells: vec![false; (width * height) as usize],
        }
    }

    #[inline]
    pub fn get(&self, x: i32, y: i32) -> bool {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return true;
        }
        self.cells[(y * self.width + x) as usize]
    }

    #[inline]
    pub fn set(&mut self, x: i32, y: i32, v: bool) {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return;
        }
        self.cells[(y * self.width + x) as usize] = v;
    }

    /// Zona libera garantita attorno allo spawn
    pub fn clear_zone(&mut self, cx: i32, cy: i32, radius: i32) {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                self.set(cx + dx, cy + dy, false);
            }
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Genera il terrain per la generazione corrente.
///
/// - `fill_rate`     : densità muri [0.20 sparso … 0.55 denso], default 0.35
/// - `blob_scale`    : dimensione cluster [2.0 grandi … 7.0 piccoli], default 4.0
/// - `smooth_passes` : passate CA di rifinitura bordi [0..3], default 1
pub fn generate(
    seed: u64,
    width: i32,
    height: i32,
    fill_rate: f32,
    blob_scale: f32,
    smooth_passes: u32,
    spawn_pos: Position,
    spawn_clearance: i32,
) -> Vec<bool> {
    let terrain_seed = derive_seed(seed, 0xDEAD_BEEF_CAFE_1234);
    let mut map = gen_clusters(
        terrain_seed,
        width,
        height,
        fill_rate,
        blob_scale,
        smooth_passes,
    );
    map.clear_zone(spawn_pos.x, spawn_pos.y, spawn_clearance);
    map.cells
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn derive_seed(base: u64, salt: u64) -> u64 {
    base.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(salt)
}

// ── fBm value noise (zero dipendenze esterne) ─────────────────────────────────
//
// Perché noise e non CA?
//
// Il CA classico parte da rumore bianco i.i.d. — ogni cella è indipendente.
// Anche con molti pass rimangono sempre celle 1×1 isolate perché non raggiungono
// mai la soglia di 5 vicini (P ≈ 0.35 per cella isolata con threshold=5).
//
// Il value noise produce direttamente un campo spazialmente correlato:
// celle vicine hanno valori simili → thresholding → cluster organici.
// Non esistono celle isolate perché il campo è continuo.
//
// Pipeline:
//   hash2()        : (seed, xi, yi) → f32[0,1]  splitmix64
//   value_noise()  : bilinear interpolation + smoothstep su 4 angoli
//   fbm()          : 3 ottave (freq×2, ampiezza×0.5 per ottava)
//   threshold      : derivato da fill_rate → celle sopra soglia = muro
//   CA leggero     : 1 pass per arrotondare i bordi (non per creare i blob)

#[inline]
fn hash2(seed: u64, xi: i32, yi: i32) -> f32 {
    let mut h = seed
        .wrapping_add((xi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((yi as u64).wrapping_mul(0x6C62_272E_07BB_0142));
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    (h >> 11) as f32 / (1u64 << 53) as f32
}

#[inline]
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn value_noise(seed: u64, x: f32, y: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = smoothstep(x - x.floor());
    let yf = smoothstep(y - y.floor());

    let v00 = hash2(seed, xi, yi);
    let v10 = hash2(seed, xi + 1, yi);
    let v01 = hash2(seed, xi, yi + 1);
    let v11 = hash2(seed, xi + 1, yi + 1);

    let lo = v00 + (v10 - v00) * xf;
    let hi = v01 + (v11 - v01) * xf;
    lo + (hi - lo) * yf
}

/// fBm: 3 ottave, lacunarity=2, gain=0.5
fn fbm(seed: u64, x: f32, y: f32, base_scale: f32) -> f32 {
    const OCTAVES: u32 = 3;
    const LACUNARITY: f32 = 2.0;
    const GAIN: f32 = 0.5;

    let mut value = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = base_scale;
    let mut max_val = 0.0f32;

    for o in 0..OCTAVES {
        let oct_seed = derive_seed(seed, o as u64 * 0x1234_5678_ABCD_EF01);
        value += value_noise(oct_seed, x * frequency, y * frequency) * amplitude;
        max_val += amplitude;
        amplitude *= GAIN;
        frequency *= LACUNARITY;
    }
    value / max_val
}

// ── Generazione cluster ───────────────────────────────────────────────────────

fn gen_clusters(
    seed: u64,
    width: i32,
    height: i32,
    fill_rate: f32,
    blob_scale: f32,
    smooth_passes: u32,
) -> TerrainMap {
    let mut map = TerrainMap::new(width, height);

    if fill_rate <= 0.0 {
        return map;
    }

    let scale = blob_scale.clamp(1.5, 10.0) / width.min(height) as f32;
    let ox = hash2(seed, 0, 0) * 100.0;
    let oy = hash2(seed, 1, 0) * 100.0;

    let mut noise_values: Vec<((i32, i32), f32)> = (1..height - 1)
        .flat_map(|y| (1..width - 1).map(move |x| (x, y)))
        .map(|(x, y)| {
            let n = fbm(seed, x as f32 * scale + ox, y as f32 * scale + oy, 1.0);
            ((x, y), n)
        })
        .collect();

    noise_values
        .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let wall_count = (noise_values.len() as f32 * fill_rate.clamp(0.0, 1.0)).round() as usize;
    for &(pos, _) in noise_values.iter().take(wall_count) {
        map.set(pos.0, pos.1, true);
    }

    for _ in 0..smooth_passes.min(3) {
        let prev = map.cells.clone();
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let walls = neighbour_wall_count(&prev, width, height, x, y);
                let is_wall = prev[(y * width + x) as usize];
                map.set(x, y, if is_wall { walls >= 3 } else { walls >= 7 });
            }
        }
    }

    map
}

fn neighbour_wall_count(cells: &[bool], width: i32, height: i32, x: i32, y: i32) -> u8 {
    let mut count = 0u8;
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || nx >= width || ny < 0 || ny >= height {
                count += 1;
            } else if cells[(ny * width + nx) as usize] {
                count += 1;
            }
        }
    }
    count
}
