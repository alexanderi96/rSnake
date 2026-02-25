//! Profiling support: pprof (flamegraph), dhat (heap), Tracy (real-time)
//!
//! Compile con:
//!   cargo run --release --features profiling       → flamegraph SVG on exit
//!   cargo run --release --features dhat-heap       → heap report JSON on exit
//!   cargo run --release --features tracy           → Tracy real-time (apri Tracy GUI prima)

// ─── dhat heap profiler ───────────────────────────────────────────────────────
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

// ─── ProfilingGuard RAII ──────────────────────────────────────────────────────

/// RAII guard that starts profiling on creation and saves reports on drop.
/// Works with any combination of profiling features.
pub struct ProfilingGuard {
    #[cfg(feature = "profiling")]
    pprof_guard: Option<pprof::ProfilerGuard<'static>>,

    #[cfg(feature = "dhat-heap")]
    _dhat_profiler: dhat::Profiler,
}

impl ProfilingGuard {
    /// Create a new profiling guard.
    /// All profiling backends are initialized if their features are enabled.
    pub fn new() -> Self {
        #[cfg(feature = "dhat-heap")]
        {
            println!("🧠 dhat heap profiler attivo — report su exit: dhat-heap.json");
        }

        #[cfg(feature = "profiling")]
        println!("🔥 pprof CPU profiler attivo — flamegraph su exit");

        #[cfg(feature = "tracy")]
        println!("📡 Tracy profiler attivo — connetti Tracy GUI ora");

        Self {
            #[cfg(feature = "profiling")]
            pprof_guard: { pprof::ProfilerGuard::new(100).ok() },

            #[cfg(feature = "dhat-heap")]
            _dhat_profiler: dhat::Profiler::new_heap(),
        }
    }
}

impl Drop for ProfilingGuard {
    fn drop(&mut self) {
        #[cfg(feature = "profiling")]
        if let Some(guard) = self.pprof_guard.take() {
            save_flamegraph(guard);
        }
        // dhat saves automatically in Drop
    }
}

#[cfg(feature = "profiling")]
fn save_flamegraph(guard: pprof::ProfilerGuard<'static>) {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let filename = format!("flamegraph-{}.svg", ts);

    match guard.report().build() {
        Ok(report) => match std::fs::File::create(&filename) {
            Ok(file) => match report.flamegraph(file) {
                Ok(_) => println!(
                    "✅ Flamegraph salvato: {}\n   Apri con: firefox {}",
                    filename, filename
                ),
                Err(e) => eprintln!("❌ Errore scrittura flamegraph: {}", e),
            },
            Err(e) => eprintln!("❌ Errore creazione file {}: {}", filename, e),
        },
        Err(e) => eprintln!("❌ Errore build report pprof: {}", e),
    }
}

/// Check if any profiling feature is enabled at compile time.
pub fn is_profiling() -> bool {
    cfg!(any(
        feature = "profiling",
        feature = "dhat-heap",
        feature = "tracy"
    ))
}
