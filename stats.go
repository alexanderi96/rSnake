package main

import (
	"encoding/json"
	"fmt"
	"os"
	"snake-game/qlearning"
	"sort"
	"sync"
	"time"
)

const (
	StatsFile = "data/stats.json"
	GroupSize = 100 // Numero di partite per gruppo
)

// GameStats contiene tutte le partite registrate e fornisce metodi per
// ottenere statistiche come punteggio medio, durata media, ecc.
type GameStats struct {
	Games []GameRecord // Array di record di partita (sia singole che raggruppate)
	mutex sync.RWMutex
}

// GameRecord rappresenta i dati di una partita (singola o raggruppata).
type GameRecord struct {
	StartTime        time.Time `json:"startTime"`
	EndTime          time.Time `json:"endTime"`
	Score            int       `json:"score"`            // Per partite singole
	CompressionIndex int       `json:"compressionIndex"` // 0 per partite singole, >0 per gruppi
	GamesCount       int       `json:"gamesCount"`       // 1 per partite singole, >1 per gruppi
	AverageScore     float64   `json:"averageScore"`     // Per gruppi
	MedianScore      float64   `json:"medianScore"`      // Per gruppi
	MaxScore         int       `json:"maxScore"`         // Per gruppi
	MinScore         int       `json:"minScore"`         // Per gruppi
	AverageDuration  float64   `json:"averageDuration"`  // Per gruppi
	MaxDuration      float64   `json:"maxDuration"`      // Per gruppi
	MinDuration      float64   `json:"minDuration"`      // Per gruppi
}

// NewGameStats crea una nuova istanza di GameStats e tenta di caricare i dati dal file.
func NewGameStats() *GameStats {
	stats := &GameStats{
		Games: make([]GameRecord, 0),
	}
	stats.loadFromFile()
	return stats
}

// AddGame aggiunge una nuova partita alle statistiche.
func (s *GameStats) AddGame(score int, startTime, endTime time.Time) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	game := GameRecord{
		StartTime:        startTime,
		EndTime:          endTime,
		Score:            score,
		CompressionIndex: 0, // Partita singola
		GamesCount:       1,
		AverageScore:     float64(score),
		MedianScore:      float64(score),
		MaxScore:         score,
		MinScore:         score,
		AverageDuration:  endTime.Sub(startTime).Seconds(),
		MaxDuration:      endTime.Sub(startTime).Seconds(),
		MinDuration:      endTime.Sub(startTime).Seconds(),
	}
	s.Games = append(s.Games, game)

	// Raggruppa le statistiche se necessario
	s.groupGames()
}

// groupGames raggruppa le partite mantenendo l'indice di compressione
func (s *GameStats) groupGames() {
	// Ordina i giochi per indice di compressione e data
	sort.Slice(s.Games, func(i, j int) bool {
		if s.Games[i].CompressionIndex != s.Games[j].CompressionIndex {
			return s.Games[i].CompressionIndex < s.Games[j].CompressionIndex
		}
		return s.Games[i].StartTime.Before(s.Games[j].StartTime)
	})

	// Trova gruppi di 10 record con lo stesso indice di compressione
	for compressionLevel := 0; ; compressionLevel++ {
		records := make([]GameRecord, 0)
		for _, game := range s.Games {
			if game.CompressionIndex == compressionLevel {
				records = append(records, game)
			}
		}

		if len(records) < GroupSize {
			break // Non abbastanza record per questo livello di compressione
		}

		// Raggruppa ogni 10 record
		var newRecords []GameRecord
		for i := 0; i < len(records); i += GroupSize {
			end := i + GroupSize
			if end > len(records) {
				// Mantieni i record rimanenti non compressi
				newRecords = append(newRecords, records[i:]...)
				break
			}

			group := records[i:end]
			var totalScore float64
			var totalDuration float64
			allScores := make([]float64, 0)
			maxScore := group[0].MaxScore
			minScore := group[0].MinScore
			maxDuration := group[0].MaxDuration
			minDuration := group[0].MinDuration
			startTime := group[0].StartTime
			endTime := group[0].EndTime
			totalGames := 0

			for _, g := range group {
				if g.MaxScore > maxScore {
					maxScore = g.MaxScore
				}
				if g.MinScore < minScore {
					minScore = g.MinScore
				}
				if g.MaxDuration > maxDuration {
					maxDuration = g.MaxDuration
				}
				if g.MinDuration < minDuration {
					minDuration = g.MinDuration
				}
				if g.StartTime.Before(startTime) {
					startTime = g.StartTime
				}
				if g.EndTime.After(endTime) {
					endTime = g.EndTime
				}
				totalScore += g.AverageScore * float64(g.GamesCount)
				totalDuration += g.AverageDuration * float64(g.GamesCount)
				totalGames += g.GamesCount
				// Aggiungi il punteggio mediano ripetuto per il numero di giochi nel gruppo
				for i := 0; i < g.GamesCount; i++ {
					allScores = append(allScores, g.MedianScore)
				}
			}

			// Calcola la mediana
			sort.Float64s(allScores)
			var medianScore float64
			if len(allScores) > 0 {
				if len(allScores)%2 == 0 {
					medianScore = (allScores[len(allScores)/2-1] + allScores[len(allScores)/2]) / 2
				} else {
					medianScore = allScores[len(allScores)/2]
				}
			}

			newRecord := GameRecord{
				StartTime:        startTime,
				EndTime:          endTime,
				CompressionIndex: compressionLevel + 1,
				GamesCount:       totalGames,
				AverageScore:     totalScore / float64(totalGames),
				MedianScore:      medianScore,
				MaxScore:         maxScore,
				MinScore:         minScore,
				AverageDuration:  totalDuration / float64(totalGames),
				MaxDuration:      maxDuration,
				MinDuration:      minDuration,
			}
			newRecords = append(newRecords, newRecord)
		}

		// Rimuovi i vecchi record compressi e aggiungi i nuovi
		remainingGames := make([]GameRecord, 0)
		for _, game := range s.Games {
			if game.CompressionIndex != compressionLevel {
				remainingGames = append(remainingGames, game)
			}
		}
		s.Games = append(remainingGames, newRecords...)
	}
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
	var totalGames int

	for _, game := range s.Games {
		totalScore += game.AverageScore * float64(game.GamesCount)
		totalGames += game.GamesCount
	}

	return totalScore / float64(totalGames)
}

// GetMedianScore calcola e restituisce il punteggio mediano.
func (s *GameStats) GetMedianScore() float64 {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if len(s.Games) == 0 {
		return 0
	}

	// Raccogli tutti i punteggi mediani pesati per il numero di giochi
	allScores := make([]float64, 0)
	for _, game := range s.Games {
		for i := 0; i < game.GamesCount; i++ {
			allScores = append(allScores, game.MedianScore)
		}
	}

	// Calcola la mediana
	sort.Float64s(allScores)
	if len(allScores) == 0 {
		return 0
	}
	if len(allScores)%2 == 0 {
		return (allScores[len(allScores)/2-1] + allScores[len(allScores)/2]) / 2
	}
	return allScores[len(allScores)/2]
}

// GetMaxScore restituisce il punteggio massimo registrato.
func (s *GameStats) GetMaxScore() int {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if len(s.Games) == 0 {
		return 0
	}

	maxScore := s.Games[0].MaxScore
	for _, game := range s.Games {
		if game.MaxScore > maxScore {
			maxScore = game.MaxScore
		}
	}

	return maxScore
}

// GetGamesPlayed restituisce il numero totale di partite giocate.
func (s *GameStats) GetGamesPlayed() int {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	total := 0
	for _, game := range s.Games {
		total += game.GamesCount
	}
	return total
}

// GetAverageDuration calcola e restituisce la durata media delle partite (in secondi).
func (s *GameStats) GetAverageDuration() float64 {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if len(s.Games) == 0 {
		return 0
	}

	var totalDuration float64
	var totalGames int

	for _, game := range s.Games {
		totalDuration += game.AverageDuration * float64(game.GamesCount)
		totalGames += game.GamesCount
	}

	return totalDuration / float64(totalGames)
}

// GetMaxDuration restituisce la durata massima di una partita (in secondi).
func (s *GameStats) GetMaxDuration() float64 {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if len(s.Games) == 0 {
		return 0
	}

	maxDuration := s.Games[0].MaxDuration
	for _, game := range s.Games {
		if game.MaxDuration > maxDuration {
			maxDuration = game.MaxDuration
		}
	}

	return maxDuration
}

// SaveToFile salva le statistiche su file in formato JSON.
func (s *GameStats) SaveToFile() error {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	// Ensure data directory exists
	if err := os.MkdirAll(qlearning.DataDir, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %v", err)
	}

	jsonData, err := json.Marshal(s.Games)
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
			return nil // Il file non esiste ancora, quindi si parte con statistiche vuote
		}
		return err
	}

	if err := json.Unmarshal(data, &s.Games); err != nil {
		return err
	}

	return nil
}
