package main

import (
	"encoding/json"
	"fmt"
	"os"
	"snake-game/qlearning"
	"sync"
	"time"
)

const (
	StatsFile = "data/stats.json"
)

// BestTimeRecord tiene traccia del miglior tempo per ogni punteggio
type BestTimeRecord struct {
	Time      float64   `json:"time"`      // Miglior tempo in secondi
	Steps     int       `json:"steps"`     // Numero di step per raggiungere il punteggio
	Timestamp time.Time `json:"timestamp"` // Quando è stato registrato questo record
}

// GameStats contiene tutte le partite registrate e fornisce metodi per
// ottenere statistiche come punteggio medio, durata media, ecc.
type GameStats struct {
	Games         []GameRecord           `json:"games"`      // Array di record di partita
	TotalGames    int                    `json:"totalGames"` // Contatore totale delle partite giocate
	TotalScore    int                    `json:"totalScore"` // Somma di tutti i punteggi
	TotalTime     float64                `json:"totalTime"`  // Somma di tutte le durate
	BestTimes     map[int]BestTimeRecord `json:"bestTimes"`  // Mappa punteggio -> miglior tempo/steps
	mutex         sync.RWMutex           `json:"-"`          // Non salvare il mutex in JSON
	lastGameTimes []time.Time            `json:"-"`          // Timestamps delle ultime partite per calcolare il rate
}

// GameRecord rappresenta i dati di una partita.
type GameRecord struct {
	StartTime     time.Time `json:"startTime"`
	EndTime       time.Time `json:"endTime"`
	Score         int       `json:"score"`
	PolicyEntropy float64   `json:"policyEntropy"` // Entropia della policy al momento della partita
}

// NewGameStats crea una nuova istanza di GameStats e tenta di caricare i dati dal file.
func NewGameStats() *GameStats {
	stats := &GameStats{
		Games:         make([]GameRecord, 0),
		TotalGames:    0,
		TotalScore:    0,
		TotalTime:     0,
		BestTimes:     make(map[int]BestTimeRecord),
		lastGameTimes: make([]time.Time, 0, 1000),
	}
	stats.loadFromFile()
	return stats
}

// GetGamesPerSecond calcola il rate attuale di partite al secondo
func (s *GameStats) GetGamesPerSecond() int {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	now := time.Now()
	cutoff := now.Add(-time.Second)

	// Rimuovi i timestamp più vecchi di 1 secondo
	validTimes := 0
	for i := len(s.lastGameTimes) - 1; i >= 0; i-- {
		if s.lastGameTimes[i].After(cutoff) {
			validTimes++
		} else {
			break
		}
	}

	return validTimes
}

// GetBestTime restituisce il miglior tempo registrato per un dato punteggio
func (s *GameStats) GetBestTime(score int) (BestTimeRecord, bool) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	record, exists := s.BestTimes[score]
	return record, exists
}

// GetTimeScalingFactor calcola un fattore di scaling per la reward basato sulla performance rispetto al miglior tempo
func (s *GameStats) GetTimeScalingFactor(score int, currentTime float64) float64 {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if bestTime, exists := s.BestTimes[score]; exists {
		if currentTime <= bestTime.Time {
			// Se è un nuovo record, massima reward
			return 1.5
		}
		// Scala la reward in base a quanto siamo vicini al record
		// 1.0 = stesso tempo del record
		// 0.5 = doppio del tempo del record
		return 1.0 / (currentTime / bestTime.Time)
	}

	// Se non abbiamo un record per questo punteggio, return 1.0 (nessuna scalatura)
	return 1.0
}

// AddGame aggiunge una nuova partita alle statistiche.
func (s *GameStats) AddGame(score int, startTime, endTime time.Time, policyEntropy float64) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	duration := endTime.Sub(startTime).Seconds()
	steps := int(duration * 10) // Approssimazione degli step basata sul tempo (assumendo 10 step/secondo)

	// Aggiorna il miglior tempo per questo punteggio se necessario
	if bestTime, exists := s.BestTimes[score]; !exists || duration < bestTime.Time {
		s.BestTimes[score] = BestTimeRecord{
			Time:      duration,
			Steps:     steps,
			Timestamp: endTime,
		}
	}

	// Incrementa i contatori totali
	s.TotalGames++

	// Aggiungi il timestamp per il calcolo del rate
	now := time.Now()
	s.lastGameTimes = append(s.lastGameTimes, now)

	// Mantieni solo i timestamp dell'ultimo secondo
	cutoff := now.Add(-time.Second)
	for i := 0; i < len(s.lastGameTimes); i++ {
		if s.lastGameTimes[i].After(cutoff) {
			s.lastGameTimes = s.lastGameTimes[i:]
			break
		}
	}
	s.TotalScore += score
	s.TotalTime += duration

	game := GameRecord{
		StartTime:     startTime,
		EndTime:       endTime,
		Score:         score,
		PolicyEntropy: policyEntropy,
	}
	s.Games = append(s.Games, game)
}

// GetStats restituisce i dati attuali delle statistiche.
func (s *GameStats) GetStats() []GameRecord {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	return s.Games
}

// GetAverageScore calcola e restituisce il punteggio medio.
func (s *GameStats) GetAverageScore() float64 {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if len(s.Games) == 0 {
		return 0
	}

	var totalScore float64
	for _, game := range s.Games {
		totalScore += float64(game.Score)
	}

	return totalScore / float64(len(s.Games))
}

// GetAbsoluteMaxScore restituisce il punteggio massimo mai registrato.
func (s *GameStats) GetAbsoluteMaxScore() int {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	maxScore := 0
	for _, game := range s.Games {
		if game.Score > maxScore {
			maxScore = game.Score
		}
	}
	return maxScore
}

// GetAverageDuration calcola e restituisce la durata media delle partite (in secondi).
func (s *GameStats) GetAverageDuration() float64 {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if len(s.Games) == 0 {
		return 0
	}

	var totalDuration float64
	for _, game := range s.Games {
		totalDuration += game.EndTime.Sub(game.StartTime).Seconds()
	}

	return totalDuration / float64(len(s.Games))
}

// SaveToFile salva le statistiche su file in formato JSON.
func (s *GameStats) SaveToFile() error {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	// Ensure data directory exists
	if err := os.MkdirAll(qlearning.DataDir, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %v", err)
	}

	jsonData, err := json.Marshal(s)
	if err != nil {
		return fmt.Errorf("failed to marshal stats data: %v", err)
	}

	if err := os.WriteFile(StatsFile, jsonData, 0644); err != nil {
		return fmt.Errorf("failed to write stats file: %v", err)
	}

	return nil
}

// loadFromFile carica le statistiche da file.
func (s *GameStats) loadFromFile() error {
	data, err := os.ReadFile(StatsFile)
	if err != nil {
		if os.IsNotExist(err) {
			s.Games = make([]GameRecord, 0)
			s.TotalGames = 0
			return nil // Il file non esiste ancora, quindi si parte con statistiche vuote
		}
		return err
	}

	if err := json.Unmarshal(data, s); err != nil {
		return err
	}

	return nil
}
