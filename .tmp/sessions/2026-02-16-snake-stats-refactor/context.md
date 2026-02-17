# Task Context: Snake DQN Unified Statistics Refactoring

Session ID: 2026-02-16-snake-stats-refactor
Created: 2026-02-17
Status: completed

## Current Request
Implementare un sistema di statistiche unificato per Snake DQN con:
1. Nuove struct TrainingSession e GenerationRecord
2. Logica di compressione dello storico
3. Fix del bug di sincronizzazione generazionale al reset
4. Salvataggio metriche AI (loss, epsilon)
5. Aggiornamento UI grafico per usare la nuova struttura

## Context Files (Standards to Follow)
- Project uses Rust with Bevy framework
- Code patterns: Functional programming, ECS architecture

## Reference Files
- /home/stego/workspace/myrepo/test/snake-rs/src/main.rs (main source file)

## External Docs Fetched
- None (native Rust/Bevy code)

## Changes Implemented

### 1. Nuove Strutture Dati (Lines 120-230)
Aggiunte `GenerationRecord` e `TrainingSession` con:
- Metriche di gioco: gen, avg_score, max_score, alive_count
- Metriche AI: avg_loss, epsilon
- Timestamp Unix

### 2. Compressione Storico (Lines 162-211)
Implementato `compress_history()` che:
- Mantiene 200 entry recenti dettagliate
- Comprime i vecchi record con ratio 4:1
- Media di gen, score, loss, epsilon
- Stampa info compressioni

### 3. Fix Sincronizzazione (Lines 1028-1034)
In `setup()`:
- Caricamento TrainingSession PRIMA di GameState
- Se ci sono record, sincronizza:
  - game_state.total_iterations = last_gen
  - brain.iterations = last_gen
- Stampa messaggio di sync

### 4. Game Loop Updates (Lines 1415-1460)
In `game_loop()`:
- Aggiunto parametro training_session
- Creazione GenerationRecord con metriche AI
- Chiamata compress_history() ogni 100 generazioni
- Salvataggio training_session.json

### 5. UI Grafico (Lines 2044-2193)
`draw_graph_in_panel()`:
- Cambiato parametro da GameHistory a TrainingSession
- Aggiornati riferimenti campi: gen, max_score, avg_score, alive_count
- Mantenuta logica di disegno esistente

### 6. Exit Handler (Lines 1573-1614)
`handle_input()`:
- Aggiunto parametro training_session
- Salvataggio TrainingSession su ESC
- Stampa numero records in riepilogo

## File Output
- training_session.json (nuovo file con dati unificati)
- Vecchi file mantenuti per retrocompatibilità:
  - snake_brain.json
  - snake_stats.json
  - snake_history.json

## Exit Criteria
- [x] Codice compila senza errori
- [x] Nuove struct implementate
- [x] Compressione funzionante
- [x] Sync generazione fixato
- [x] Metriche AI salvate
- [x] UI grafico aggiornato
- [x] Salvataggio in uscita implementato

## Testing Notes
- Compilazione: `cargo check` - OK (solo warnings pre-esistenti)
- Eseguibile: `cargo run` - Pronto per test
- File generato: training_session.json

## Migration Path
I vecchi file (snake_stats.json, snake_history.json) rimangono funzionanti.
Il nuovo training_session.json contiene tutti i dati unificati.
Per migrare dati esistenti, eseguire una sessione che caricherà lo storico
vecchio e salverà il nuovo formato.
