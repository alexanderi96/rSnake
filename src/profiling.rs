//! Profiling support for snake-rs
//!
//! To enable profiling, compile with:
//!   cargo run --features profiling
//!
//! This will generate a flamegraph at: profile-{timestamp}.svg

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

static PROFILER_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Start CPU profiling
/// Call this at the beginning of your main function
pub fn start_profiling() -> Option<pprof::ProfilerGuard<'static>> {
    if PROFILER_ACTIVE.swap(true, Ordering::SeqCst) {
        eprintln!("⚠️ Profiling already active");
        return None;
    }

    println!("🔥 CPU Profiling started - will save flamegraph on exit");
    println!("   Compile with: cargo run --features profiling");

    // Start profiler with 100Hz sampling (Go-like default)
    match pprof::ProfilerGuard::new(100) {
        Ok(guard) => Some(guard),
        Err(e) => {
            eprintln!("❌ Failed to start profiler: {}", e);
            PROFILER_ACTIVE.store(false, Ordering::SeqCst);
            None
        }
    }
}

/// Stop profiling and save flamegraph
/// Call this before exit (or in Drop implementation)
pub fn stop_profiling(guard: Option<pprof::ProfilerGuard>) {
    if guard.is_none() {
        return;
    }

    let timestamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let filename = format!("profile-{}.svg", timestamp);

    println!(
        "🔥 Stopping profiler and generating flamegraph: {}",
        filename
    );

    match guard.unwrap().report().build() {
        Ok(report) => {
            // Generate flamegraph SVG
            let file = std::fs::File::create(&filename).expect("Failed to create profile file");
            match report.flamegraph(file) {
                Ok(_) => {
                    println!("✅ Flamegraph saved to: {}", filename);
                    println!("   Open in browser: firefox {}\n", filename);
                }
                Err(e) => eprintln!("❌ Failed to write flamegraph: {}", e),
            }

            // Also print top functions to console (Go pprof style)
            println!("\n📊 Profiling complete - view flamegraph for details");
            println!("   File: {}", filename);
            println!();
        }
        Err(e) => eprintln!("❌ Failed to build report: {}", e),
    }

    PROFILER_ACTIVE.store(false, Ordering::SeqCst);
}

/// RAII guard for automatic profiling stop
pub struct ProfilingGuard(Option<pprof::ProfilerGuard<'static>>);

impl ProfilingGuard {
    pub fn new() -> Self {
        Self(start_profiling())
    }
}

impl Drop for ProfilingGuard {
    fn drop(&mut self) {
        stop_profiling(self.0.take());
    }
}

/// Check if profiling is active
pub fn is_profiling() -> bool {
    PROFILER_ACTIVE.load(Ordering::SeqCst)
}
