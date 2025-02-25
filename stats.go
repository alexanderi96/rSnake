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
	GroupSize = 10 // Numero di partite per gruppo
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

// groupGames raggruppa le partite in gruppi di GroupSize
func (s *GameStats) groupGames() {
	// Ordina tutti i record per data (dal più vecchio al più nuovo)
	sort.Slice(s.Games, func(i, j int) bool {
		return s.Games[i].StartTime.Before(s.Games[j].StartTime)
	})

	// Se non abbiamo abbastanza record, termina
	if len(s.Games) <= GroupSize {
		return
	}

	// Cerca il record attivo che sta comprimendo
	activeIdx := -1
	for i, game := range s.Games {
		if game.CompressionIndex > 0 && game.GamesCount < GroupSize {
			activeIdx = i
			break
		}
	}

	// Se non c'è un record attivo
	if activeIdx == -1 {
		// Cerca il primo record che ha raggiunto GroupSize elementi
		for i, game := range s.Games {
			if game.GamesCount >= GroupSize && i+1 < len(s.Games) && s.Games[i+1].CompressionIndex == 0 {
				// Inizia la compressione del record successivo
				nextRecord := s.Games[i+1]
				nextRecord.CompressionIndex = game.CompressionIndex
				s.Games[i+1] = nextRecord
				return
			}
		}

		// Se non abbiamo trovato un record pieno, ma abbiamo più di GroupSize record non compressi
		uncompressedCount := 0
		for _, game := range s.Games {
			if game.CompressionIndex == 0 {
				uncompressedCount++
			}
		}

		if uncompressedCount > GroupSize {
			// Trova il primo record non compresso
			for i, game := range s.Games {
				if game.CompressionIndex == 0 {
					game.CompressionIndex = 1
					s.Games[i] = game
					return
				}
			}
		}
	} else {
		// Continua la compressione del record attivo
		if activeIdx+1 < len(s.Games) && s.Games[activeIdx+1].CompressionIndex == 0 {
			s.mergeRecords(activeIdx, activeIdx+1)
		}
	}
}

// mergeRecords unisce due record adiacenti, dove il primo è già compresso
func (s *GameStats) mergeRecords(compressedIdx, newIdx int) {
	if compressedIdx >= len(s.Games) || newIdx >= len(s.Games) {
		return
	}

	compressed := s.Games[compressedIdx]
	newRecord := s.Games[newIdx]

	// Calcola il nuovo numero totale di giochi
	totalGames := compressed.GamesCount + 1

	// Calcola le nuove statistiche
	newAvgScore := (compressed.AverageScore*float64(compressed.GamesCount) + float64(newRecord.Score)) / float64(totalGames)

	// Aggiorna min/max score
	newMaxScore := compressed.MaxScore
	if newRecord.Score > newMaxScore {
		newMaxScore = newRecord.Score
	}
	newMinScore := compressed.MinScore
	if newRecord.Score < newMinScore {
		newMinScore = newRecord.Score
	}

	// Calcola la nuova durata media e min/max
	newDuration := newRecord.EndTime.Sub(newRecord.StartTime).Seconds()
	newAvgDuration := (compressed.AverageDuration*float64(compressed.GamesCount) + newDuration) / float64(totalGames)
	newMaxDuration := compressed.MaxDuration
	if newDuration > newMaxDuration {
		newMaxDuration = newDuration
	}
	newMinDuration := compressed.MinDuration
	if newDuration < newMinDuration {
		newMinDuration = newDuration
	}

	// Calcola la nuova mediana
	scores := make([]float64, 0, totalGames)
	// Aggiungi i punteggi del record compresso
	for i := 0; i < compressed.GamesCount; i++ {
		scores = append(scores, compressed.MedianScore)
	}
	// Aggiungi il nuovo punteggio
	scores = append(scores, float64(newRecord.Score))
	sort.Float64s(scores)
	var newMedianScore float64
	if len(scores)%2 == 0 {
		newMedianScore = (scores[len(scores)/2-1] + scores[len(scores)/2]) / 2
	} else {
		newMedianScore = scores[len(scores)/2]
	}

	// Aggiorna il record compresso
	compressed.GamesCount = totalGames
	compressed.AverageScore = newAvgScore
	compressed.MedianScore = newMedianScore
	compressed.MaxScore = newMaxScore
	compressed.MinScore = newMinScore
	compressed.AverageDuration = newAvgDuration
	compressed.MaxDuration = newMaxDuration
	compressed.MinDuration = newMinDuration
	if newRecord.EndTime.After(compressed.EndTime) {
		compressed.EndTime = newRecord.EndTime
	}

	// Rimuovi il record assorbito e aggiorna il record compresso
	newGames := make([]GameRecord, 0, len(s.Games)-1)
	newGames = append(newGames, s.Games[:compressedIdx]...)
	newGames = append(newGames, compressed)
	newGames = append(newGames, s.Games[newIdx+1:]...)
	s.Games = newGames
}

// GetStats restituisce i dati attuali delle statistiche.
func (s *GameStats) GetStats() []GameRecord {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	return s.Games
}

// GetStatsForLevel restituisce le statistiche per un determinato livello di compressione
func (s *GameStats) GetStatsForLevel(compressionLevel int) []GameRecord {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	records := make([]GameRecord, 0)
	for _, game := range s.Games {
		if game.CompressionIndex == compressionLevel {
			records = append(records, game)
		}
	}
	return records
}

// GetAverageScore calcola e restituisce il punteggio medio per un determinato livello di compressione.
func (s *GameStats) GetAverageScore(compressionLevel int) float64 {
	records := s.GetStatsForLevel(compressionLevel)
	if len(records) == 0 {
		return 0
	}

	var totalScore float64
	var totalGames int

	for _, game := range records {
		totalScore += game.AverageScore * float64(game.GamesCount)
		totalGames += game.GamesCount
	}

	return totalScore / float64(totalGames)
}

// GetMedianScore calcola e restituisce il punteggio mediano per un determinato livello di compressione.
func (s *GameStats) GetMedianScore(compressionLevel int) float64 {
	records := s.GetStatsForLevel(compressionLevel)
	if len(records) == 0 {
		return 0
	}

	// Raccogli tutti i punteggi mediani pesati per il numero di giochi
	allScores := make([]float64, 0)
	for _, game := range records {
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

// GetMaxCompressionLevel restituisce il massimo livello di compressione presente nelle statistiche
func (s *GameStats) GetMaxCompressionLevel() int {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	maxLevel := 0
	for _, game := range s.Games {
		if game.CompressionIndex > maxLevel {
			maxLevel = game.CompressionIndex
		}
	}
	return maxLevel
}

// GetMaxScore restituisce il punteggio massimo registrato per un determinato livello di compressione.
func (s *GameStats) GetMaxScore(compressionLevel int) int {
	records := s.GetStatsForLevel(compressionLevel)
	if len(records) == 0 {
		return 0
	}

	maxScore := records[0].MaxScore
	for _, game := range records {
		if game.MaxScore > maxScore {
			maxScore = game.MaxScore
		}
	}

	return maxScore
}

// GetGamesPlayed restituisce il numero totale di partite giocate per un determinato livello di compressione.
func (s *GameStats) GetGamesPlayed(compressionLevel int) int {
	records := s.GetStatsForLevel(compressionLevel)

	total := 0
	for _, game := range records {
		total += game.GamesCount
	}
	return total
}

// GetAverageDuration calcola e restituisce la durata media delle partite (in secondi) per un determinato livello di compressione.
func (s *GameStats) GetAverageDuration(compressionLevel int) float64 {
	records := s.GetStatsForLevel(compressionLevel)
	if len(records) == 0 {
		return 0
	}

	var totalDuration float64
	var totalGames int

	for _, game := range records {
		totalDuration += game.AverageDuration * float64(game.GamesCount)
		totalGames += game.GamesCount
	}

	return totalDuration / float64(totalGames)
}

// GetMaxDuration restituisce la durata massima di una partita (in secondi) per un determinato livello di compressione.
func (s *GameStats) GetMaxDuration(compressionLevel int) float64 {
	records := s.GetStatsForLevel(compressionLevel)
	if len(records) == 0 {
		return 0
	}

	maxDuration := records[0].MaxDuration
	for _, game := range records {
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
