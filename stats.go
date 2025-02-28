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
	Games         []GameRecord `json:"games"`      // Array di record di partita (sia singole che raggruppate)
	TotalGames    int          `json:"totalGames"` // Contatore totale delle partite giocate
	TotalScore    int          `json:"totalScore"` // Somma di tutti i punteggi
	TotalTime     float64      `json:"totalTime"`  // Somma di tutte le durate
	mutex         sync.RWMutex `json:"-"`          // Non salvare il mutex in JSON
	lastGameTimes []time.Time  `json:"-"`          // Timestamps delle ultime partite per calcolare il rate
}

// GameRecord rappresenta i dati di una partita (singola o raggruppata).
type GameRecord struct {
	StartTime                 time.Time `json:"startTime"`
	EndTime                   time.Time `json:"endTime"`
	Score                     int       `json:"score"`                     // Per partite singole
	CompressionIndex          int       `json:"compressionIndex"`          // 0 per partite singole, >0 per gruppi
	GamesCount                int       `json:"gamesCount"`                // 1 per partite singole, >1 per gruppi
	CompressedAverageScore    float64   `json:"compressedAverageScore"`    // Media del gruppo compresso
	MaxScore                  int       `json:"maxScore"`                  // Per gruppi
	MinScore                  int       `json:"minScore"`                  // Per gruppi
	CompressedAverageDuration float64   `json:"compressedAverageDuration"` // Media del gruppo compresso
	MaxDuration               float64   `json:"maxDuration"`               // Per gruppi
	MinDuration               float64   `json:"minDuration"`               // Per gruppi
	AverageMaxScore           float64   `json:"averageMaxScore"`           // Media dei punteggi massimi
	AverageMinScore           float64   `json:"averageMinScore"`           // Media dei punteggi minimi
	AverageMaxDuration        float64   `json:"averageMaxDuration"`        // Media delle durate massime
	AverageMinDuration        float64   `json:"averageMinDuration"`        // Media delle durate minime
	Epsilon                   float64   `json:"epsilon"`                   // Valore epsilon al momento della partita
}

// NewGameStats crea una nuova istanza di GameStats e tenta di caricare i dati dal file.
func NewGameStats() *GameStats {
	stats := &GameStats{
		Games:         make([]GameRecord, 0),
		TotalGames:    0,
		TotalScore:    0,
		TotalTime:     0,
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

// AddGame aggiunge una nuova partita alle statistiche.
func (s *GameStats) AddGame(score int, startTime, endTime time.Time, epsilon float64) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

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
	duration := endTime.Sub(startTime).Seconds()
	s.TotalTime += duration

	game := GameRecord{
		StartTime:                 startTime,
		EndTime:                   endTime,
		Score:                     score,
		CompressionIndex:          0, // Partita singola
		GamesCount:                1,
		CompressedAverageScore:    float64(score),
		MaxScore:                  score,
		MinScore:                  score,
		CompressedAverageDuration: duration,
		MaxDuration:               duration,
		MinDuration:               duration,
		AverageMaxScore:           float64(score),
		AverageMinScore:           float64(score),
		AverageMaxDuration:        duration,
		AverageMinDuration:        duration,
		Epsilon:                   epsilon,
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
	newCompressedAvgScore := (compressed.CompressedAverageScore*float64(compressed.GamesCount) + float64(newRecord.Score)) / float64(totalGames)

	// Per il record compresso, usa i valori esistenti
	newMaxScore := compressed.MaxScore
	newMinScore := compressed.MinScore

	// Per il nuovo record, confronta con Score
	if newRecord.GamesCount == 1 {
		if newRecord.Score > newMaxScore {
			newMaxScore = newRecord.Score
		}
		if newRecord.Score < newMinScore {
			newMinScore = newRecord.Score
		}
	} else {
		// Se il nuovo record è già compresso, confronta con i suoi max/min
		if newRecord.MaxScore > newMaxScore {
			newMaxScore = newRecord.MaxScore
		}
		if newRecord.MinScore < newMinScore {
			newMinScore = newRecord.MinScore
		}
	}

	// Calcola la nuova durata media e min/max
	var newDuration float64
	if newRecord.GamesCount == 1 {
		newDuration = newRecord.EndTime.Sub(newRecord.StartTime).Seconds()
	} else {
		// Se il nuovo record è già compresso, usa la sua durata media compressa
		newDuration = newRecord.CompressedAverageDuration
	}

	newCompressedAvgDuration := (compressed.CompressedAverageDuration*float64(compressed.GamesCount) + newDuration) / float64(totalGames)

	// Aggiorna max/min duration considerando se il record è già compresso
	newMaxDuration := compressed.MaxDuration
	newMinDuration := compressed.MinDuration

	if newRecord.GamesCount == 1 {
		if newDuration > newMaxDuration {
			newMaxDuration = newDuration
		}
		if newDuration < newMinDuration {
			newMinDuration = newDuration
		}
	} else {
		if newRecord.MaxDuration > newMaxDuration {
			newMaxDuration = newRecord.MaxDuration
		}
		if newRecord.MinDuration < newMinDuration {
			newMinDuration = newRecord.MinDuration
		}
	}

	// Aggiorna il record compresso
	compressed.GamesCount = totalGames
	compressed.CompressedAverageScore = newCompressedAvgScore
	compressed.MaxScore = newMaxScore
	compressed.MinScore = newMinScore
	compressed.CompressedAverageDuration = newCompressedAvgDuration
	compressed.MaxDuration = newMaxDuration
	compressed.MinDuration = newMinDuration
	// Calcola le medie dei massimi e minimi
	compressed.AverageMaxScore = (compressed.AverageMaxScore*float64(compressed.GamesCount-1) + float64(newMaxScore)) / float64(totalGames)
	compressed.AverageMinScore = (compressed.AverageMinScore*float64(compressed.GamesCount-1) + float64(newMinScore)) / float64(totalGames)
	compressed.AverageMaxDuration = (compressed.AverageMaxDuration*float64(compressed.GamesCount-1) + newMaxDuration) / float64(totalGames)
	compressed.AverageMinDuration = (compressed.AverageMinDuration*float64(compressed.GamesCount-1) + newMinDuration) / float64(totalGames)
	// Calcola la media dell'epsilon
	compressed.Epsilon = (compressed.Epsilon*float64(compressed.GamesCount-1) + newRecord.Epsilon) / float64(totalGames)

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
		if game.CompressionIndex == 0 {
			totalScore += float64(game.Score)
			totalGames++
		} else {
			totalScore += game.CompressedAverageScore * float64(game.GamesCount)
			totalGames += game.GamesCount
		}
	}

	return totalScore / float64(totalGames)
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

// GetEpisodesSinceLastMaxScore calcola il numero di episodi trascorsi dall'ultima volta che si è ottenuto un nuovo punteggio massimo
func (s *GameStats) GetEpisodesSinceLastMaxScore() int {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if len(s.Games) == 0 {
		return 0
	}

	maxScore := 0
	episodesSinceMax := 0

	// Prima troviamo il punteggio massimo
	for _, game := range s.Games {
		currentScore := 0
		if game.GamesCount == 1 {
			currentScore = game.Score
		} else {
			currentScore = game.MaxScore
		}
		if currentScore > maxScore {
			maxScore = currentScore
		}
	}

	// Ora contiamo gli episodi dopo la prima volta che abbiamo raggiunto il massimo
	foundMax := false
	for _, game := range s.Games {
		currentScore := 0
		if game.GamesCount == 1 {
			currentScore = game.Score
		} else {
			currentScore = game.MaxScore
		}

		if !foundMax {
			if currentScore == maxScore {
				foundMax = true
			}
		} else {
			episodesSinceMax += game.GamesCount
		}
	}

	return episodesSinceMax
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
		if game.CompressionIndex == 0 {
			totalDuration += game.EndTime.Sub(game.StartTime).Seconds()
			totalGames++
		} else {
			totalDuration += game.CompressedAverageDuration * float64(game.GamesCount)
			totalGames += game.GamesCount
		}
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
