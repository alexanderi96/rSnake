# Profiling Guide for snake-rs

Questa guida ti aiuta a identificare i colli di bottiglia nelle performance usando strumenti simili a quelli di Go.

## 🔥 Quick Start

### Metodo 1: Script Automatico (Consigliato)

```bash
# Profila per 30 secondi (default)
./profile.sh

# Profila per 60 secondi
./profile.sh 60
```

Lo script:
1. Compila con feature `profiling`
2. Avvia il gioco per N secondi
3. Genera automaticamente il flamegraph SVG
4. Mostra come visualizzarlo

### Metodo 2: Manuale

```bash
# 1. Compila con profiling
cargo build --release --features profiling

# 2. Esegui (genera profile-*.svg alla chiusura)
./target/release/snake-rs

# 3. Premi 'T' per turbo mode (training senza rendering)
# 4. Lascia girare per 30-60 secondi
# 5. Chiudi con 'Q' o Ctrl+C

# 6. Apri il flamegraph
firefox profile-*.svg
```

## 📊 Come Leggere il Flamegraph

Il flamegraph è un grafico a fiamma dove:

- **Larghezza** = Tempo totale speso in quella funzione
- **Altezza** = Profondità dello stack di chiamate
- **Colori** = Casuali (solo per distinguere le funzioni)

### Pattern da cercare:

1. **Plataforme larghe** = Funzioni che consumano molto tempo
2. **Torri alte** = Call stack profonde (possibile over-engineering)
3. **Funzioni ripetute** = Chiamate frequenti (possibile caching opportunity)

### Esempio:

```
┌─────────────────────────────────────┐
│         run_simulation_step         │ ← Root function
├───────────────┬─────────────────────┤
│  agent_batch  │   update_positions  │ ← 50% | 30%
├───────────────┤                     │
│ forward_batch │                     │ ← 45% (bottleneck!)
└───────────────┴─────────────────────┘
```

## 🎯 Ottimizzazioni Comuni

### Se vedi `forward_batch` lento:
- ✅ Batch inference è già implementato
- ⚠️ Prova backend CPU (NdArray) invece di WGPU per batch piccoli

### Se vedi `run_simulation_step` lento:
- ✅ Parallelizzazione stati con Rayon già implementata
- ⚠️ Verifica che `select_actions_batch` sia usato (non loop sequenziale)

### Se vedi `rayon::` nelle prime posizioni:
- ✅ Normale - parallelizzazione sta lavorando
- ⚠️ Se troppo tempo, riduci numero di thread: `RAYON_NUM_THREADS=8 cargo run`

## 🔧 Opzioni Avanzate

### Profiling senza GUI (headless)

Modifica temporaneamente `main.rs` per disabilitare il rendering:

```rust
// In main(), aggiungi prima di App::new():
std::env::set_var("RUST_LOG", "warn");
```

### Profiling specifico di una funzione

Aggiungi marker manuali nel codice:

```rust
#[cfg(feature = "profiling")]
{
    println!("⏱️  Starting expensive operation...");
    let start = std::time::Instant::now();
    
    // ... codice da profilare ...
    
    println!("⏱️  Completed in {:?}", start.elapsed());
}
```

### Usare `cargo flamegraph` (alternativa)

Se preferisci non usare la feature integrata:

```bash
# Installa cargo-flamegraph
cargo install flamegraph

# Esegui (richiede sudo per profiling)
sudo cargo flamegraph --release

# Output: flamegraph.svg
```

**Vantaggi:**
- Non richiede modifiche al codice
- Più preciso per profiling di sistema

**Svantaggi:**
- Richiede sudo/root
- Non funziona sempre con GPU/WGPU

## 📈 Interpretazione Risultati

### Buono:
- `forward_batch` occupa ~60-70% del tempo (GPU bound, normale)
- `par_iter` occupa ~10-20% (parallelizzazione funziona)
- `update_positions` occupa ~5-10% (logica gioco)

### Da ottimizzare:
- `forward_single` o `select_action` ripetuti (inference sequenziale)
- `clone()` su tensori grandi
- `Vec::push` in loop caldi

## 🆘 Troubleshooting

### "No profile file generated"
- Assicurati di chiudere il programma (premi 'Q')
- Verifica che il programma giri per almeno 5-10 secondi

### "Permission denied" con cargo-flamegraph
- Richiede capabilities per profiling: `sudo setcap cap_perfmon+ep ./target/release/snake-rs`
- O usa: `sudo cargo flamegraph`

### Flamegraph vuoto o poco popolato
- Esegui in modalità turbo ('T') per più azioni
- Aumenta durata profiling (60+ secondi)
- Verifica che il training stia effettivamente girando

## 🔗 Risorse

- [pprof-rs documentation](https://github.com/tikv/pprof-rs)
- [Flamegraph interpretation guide](https://www.brendangregg.com/flamegraphs.html)
- [Rust profiling workshop](https://github.com/nrc/profiling-rust-workshop)

---

**Happy profiling!** 🚀
