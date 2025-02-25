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
	Games      []GameRecord `json:"games"`      // Array di record di partita (sia singole che raggruppate)
	TotalGames int          `json:"totalGames"` // Contatore totale delle partite giocate
	mutex      sync.RWMutex `json:"-"`          // Non salvare il mutex in JSON
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
		Games:      make([]GameRecord, 0),
		TotalGames: 0,
	}
	stats.loadFromFile()
	return stats
}

// AddGame aggiunge una nuova partita alle statistiche.
func (s *GameStats) AddGame(score int, startTime, endTime time.Time) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	// Incrementa il contatore totale delle partite
	s.TotalGames++

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

	// Ottieni il massimo livello di compressione attuale
	maxLevel := 0
	for _, game := range s.Games {
		if game.CompressionIndex > maxLevel {
			maxLevel = game.CompressionIndex
		}
	}

	// Per ogni livello, controlla se ci sono più di GroupSize elementi consecutivi
	for level := 0; level <= maxLevel; level++ {
		for {
			// Conta quanti elementi abbiamo per questo livello
			sameLevel := make([]int, 0) // Indici dei record con lo stesso livello
			for i := 0; i < len(s.Games); i++ {
				if s.Games[i].CompressionIndex == level {
					sameLevel = append(sameLevel, i)
				}
			}

			// Se non abbiamo più di GroupSize elementi, passa al livello successivo
			if len(sameLevel) <= GroupSize {
				break
			}

			// Promuovi il record più vecchio al livello successivo
			oldestIdx := sameLevel[0]

			// Cerca se esiste già un record del livello successivo che può assorbire
			nextLevelExists := false
			nextLevelIdx := -1
			for i := 0; i < oldestIdx; i++ {
				if s.Games[i].CompressionIndex == level+1 && s.Games[i].GamesCount < GroupSize {
					nextLevelExists = true
					nextLevelIdx = i
					break
				}
			}

			if nextLevelExists {
				// Prova ad assorbire nel record esistente
				if !s.mergeRecords(nextLevelIdx, oldestIdx) {
					// Se il merge fallisce, promuovi al livello successivo e resetta il conteggio
					s.Games[oldestIdx].CompressionIndex = level + 1
					s.Games[oldestIdx].GamesCount = 1
				}
			} else {
				// Promuovi direttamente al livello successivo e resetta il conteggio
				s.Games[oldestIdx].CompressionIndex = level + 1
				s.Games[oldestIdx].GamesCount = 1
			}

			// Aggiorna il massimo livello se necessario
			if level+1 > maxLevel {
				maxLevel = level + 1
			}
		}
	}
}

// mergeRecords unisce due record adiacenti, dove il primo è già compresso
// Restituisce true se il merge è avvenuto, false se il record compresso ha raggiunto il limite
func (s *GameStats) mergeRecords(compressedIdx, newIdx int) bool {
	if compressedIdx >= len(s.Games) || newIdx >= len(s.Games) {
		return false
	}

	compressed := s.Games[compressedIdx]
	newRecord := s.Games[newIdx]

	// Se il record compresso ha già GroupSize elementi, non può assorbire altri
	if compressed.GamesCount >= GroupSize {
		return false
	}

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

	return true
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
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	maxScore := 0
	for _, game := range s.Games {
		if game.CompressionIndex == compressionLevel {
			// Per record non compressi usa Score, per quelli compressi usa MaxScore
			if game.GamesCount == 1 {
				if game.Score > maxScore {
					maxScore = game.Score
				}
			} else {
				if game.MaxScore > maxScore {
					maxScore = game.MaxScore
				}
			}
		}
	}
	return maxScore
}

// GetAbsoluteMaxScore restituisce il punteggio massimo mai registrato tra tutti i livelli di compressione.
func (s *GameStats) GetAbsoluteMaxScore() int {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	maxScore := 0
	for _, game := range s.Games {
		// Per record non compressi usa Score, per quelli compressi usa MaxScore
		if game.GamesCount == 1 {
			if game.Score > maxScore {
				maxScore = game.Score
			}
		} else {
			if game.MaxScore > maxScore {
				maxScore = game.MaxScore
			}
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
