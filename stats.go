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
// BestTimeRecord tiene traccia del miglior tempo per ogni punteggio
type BestTimeRecord struct {
	Time      float64   `json:"time"`      // Miglior tempo in secondi
	Steps     int       `json:"steps"`     // Numero di step per raggiungere il punteggio
	Timestamp time.Time `json:"timestamp"` // Quando è stato registrato questo record
}

type GameStats struct {
	Games         []GameRecord           `json:"games"`      // Array di record di partita (sia singole che raggruppate)
	TotalGames    int                    `json:"totalGames"` // Contatore totale delle partite giocate
	TotalScore    int                    `json:"totalScore"` // Somma di tutti i punteggi
	TotalTime     float64                `json:"totalTime"`  // Somma di tutte le durate
	BestTimes     map[int]BestTimeRecord `json:"bestTimes"`  // Mappa punteggio -> miglior tempo/steps
	mutex         sync.RWMutex           `json:"-"`          // Non salvare il mutex in JSON
	lastGameTimes []time.Time            `json:"-"`          // Timestamps delle ultime partite per calcolare il rate
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
	PolicyEntropy             float64   `json:"policyEntropy"`             // Entropia della policy al momento della partita
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
		PolicyEntropy:             policyEntropy,
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

	// Per ogni livello, controlla se ci sono abbastanza elementi da comprimere
	for level := 0; level <= maxLevel+1; level++ {
		for {
			// Trova tutti i record del livello corrente
			sameLevel := make([]int, 0)
			for i := 0; i < len(s.Games); i++ {
				if s.Games[i].CompressionIndex == level {
					sameLevel = append(sameLevel, i)
				}
			}

			// Se non abbiamo abbastanza record per formare un gruppo, passa al livello successivo
			if len(sameLevel) < GroupSize {
				break
			}

			// Calcola quanti gruppi completi possiamo formare
			numGroups := len(sameLevel) / GroupSize

			// Per ogni gruppo completo
			for g := 0; g < numGroups; g++ {
				startIdx := g * GroupSize
				endIdx := startIdx + GroupSize

				// Crea un nuovo record compresso
				baseRecord := s.Games[sameLevel[startIdx]]
				newRecord := GameRecord{
					StartTime:        baseRecord.StartTime,
					EndTime:          s.Games[sameLevel[endIdx-1]].EndTime,
					CompressionIndex: level + 1,
					GamesCount:       0,
				}

				// Inizializza i valori con il primo record
				if baseRecord.GamesCount == 1 {
					newRecord.MaxScore = baseRecord.Score
					newRecord.MinScore = baseRecord.Score
					newRecord.MaxDuration = baseRecord.EndTime.Sub(baseRecord.StartTime).Seconds()
					newRecord.MinDuration = newRecord.MaxDuration
					newRecord.CompressedAverageScore = float64(baseRecord.Score)
					newRecord.CompressedAverageDuration = newRecord.MaxDuration
					newRecord.AverageMaxScore = float64(baseRecord.Score)
					newRecord.AverageMinScore = float64(baseRecord.Score)
					newRecord.AverageMaxDuration = newRecord.MaxDuration
					newRecord.AverageMinDuration = newRecord.MaxDuration
					newRecord.PolicyEntropy = baseRecord.PolicyEntropy
					newRecord.GamesCount = 1
				} else {
					newRecord.MaxScore = baseRecord.MaxScore
					newRecord.MinScore = baseRecord.MinScore
					newRecord.MaxDuration = baseRecord.MaxDuration
					newRecord.MinDuration = baseRecord.MinDuration
					newRecord.CompressedAverageScore = baseRecord.CompressedAverageScore
					newRecord.CompressedAverageDuration = baseRecord.CompressedAverageDuration
					newRecord.AverageMaxScore = baseRecord.AverageMaxScore
					newRecord.AverageMinScore = baseRecord.AverageMinScore
					newRecord.AverageMaxDuration = baseRecord.AverageMaxDuration
					newRecord.AverageMinDuration = baseRecord.AverageMinDuration
					newRecord.PolicyEntropy = baseRecord.PolicyEntropy
					newRecord.GamesCount = baseRecord.GamesCount
				}

				// Aggrega i valori dei record rimanenti nel gruppo
				for i := startIdx + 1; i < endIdx; i++ {
					record := s.Games[sameLevel[i]]
					games := record.GamesCount
					if games == 1 {
						// Record singolo
						score := float64(record.Score)
						duration := record.EndTime.Sub(record.StartTime).Seconds()

						// Aggiorna max/min
						if record.Score > newRecord.MaxScore {
							newRecord.MaxScore = record.Score
						}
						if record.Score < newRecord.MinScore {
							newRecord.MinScore = record.Score
						}
						if duration > newRecord.MaxDuration {
							newRecord.MaxDuration = duration
						}
						if duration < newRecord.MinDuration {
							newRecord.MinDuration = duration
						}

						// Aggiorna medie pesate
						totalGames := newRecord.GamesCount + 1
						newRecord.CompressedAverageScore = (newRecord.CompressedAverageScore*float64(newRecord.GamesCount) + score) / float64(totalGames)
						newRecord.CompressedAverageDuration = (newRecord.CompressedAverageDuration*float64(newRecord.GamesCount) + duration) / float64(totalGames)
						newRecord.AverageMaxScore = (newRecord.AverageMaxScore*float64(newRecord.GamesCount) + score) / float64(totalGames)
						newRecord.AverageMinScore = (newRecord.AverageMinScore*float64(newRecord.GamesCount) + score) / float64(totalGames)
						newRecord.AverageMaxDuration = (newRecord.AverageMaxDuration*float64(newRecord.GamesCount) + duration) / float64(totalGames)
						newRecord.AverageMinDuration = (newRecord.AverageMinDuration*float64(newRecord.GamesCount) + duration) / float64(totalGames)
						newRecord.PolicyEntropy = (newRecord.PolicyEntropy*float64(newRecord.GamesCount) + record.PolicyEntropy) / float64(totalGames)
						newRecord.GamesCount++
					} else {
						// Record già compresso
						totalGames := newRecord.GamesCount + games

						// Aggiorna max/min
						if record.MaxScore > newRecord.MaxScore {
							newRecord.MaxScore = record.MaxScore
						}
						if record.MinScore < newRecord.MinScore {
							newRecord.MinScore = record.MinScore
						}
						if record.MaxDuration > newRecord.MaxDuration {
							newRecord.MaxDuration = record.MaxDuration
						}
						if record.MinDuration < newRecord.MinDuration {
							newRecord.MinDuration = record.MinDuration
						}

						// Aggiorna medie pesate
						newRecord.CompressedAverageScore = (newRecord.CompressedAverageScore*float64(newRecord.GamesCount) +
							record.CompressedAverageScore*float64(games)) / float64(totalGames)
						newRecord.CompressedAverageDuration = (newRecord.CompressedAverageDuration*float64(newRecord.GamesCount) +
							record.CompressedAverageDuration*float64(games)) / float64(totalGames)
						newRecord.AverageMaxScore = (newRecord.AverageMaxScore*float64(newRecord.GamesCount) +
							record.AverageMaxScore*float64(games)) / float64(totalGames)
						newRecord.AverageMinScore = (newRecord.AverageMinScore*float64(newRecord.GamesCount) +
							record.AverageMinScore*float64(games)) / float64(totalGames)
						newRecord.AverageMaxDuration = (newRecord.AverageMaxDuration*float64(newRecord.GamesCount) +
							record.AverageMaxDuration*float64(games)) / float64(totalGames)
						newRecord.AverageMinDuration = (newRecord.AverageMinDuration*float64(newRecord.GamesCount) +
							record.AverageMinDuration*float64(games)) / float64(totalGames)
						newRecord.PolicyEntropy = (newRecord.PolicyEntropy*float64(newRecord.GamesCount) +
							record.PolicyEntropy*float64(games)) / float64(totalGames)
						newRecord.GamesCount += games
					}
				}

				// Rimuovi i record del gruppo e inserisci il nuovo record compresso
				newGames := make([]GameRecord, 0, len(s.Games)-GroupSize+1)
				newGames = append(newGames, s.Games[:sameLevel[startIdx]]...)
				newGames = append(newGames, newRecord)
				if endIdx < len(sameLevel) {
					newGames = append(newGames, s.Games[sameLevel[startIdx+1]:sameLevel[endIdx]]...)
					newGames = append(newGames, s.Games[sameLevel[endIdx]:]...)
				} else {
					newGames = append(newGames, s.Games[sameLevel[endIdx-1]+1:]...)
				}
				s.Games = newGames

				// Aggiorna gli indici dopo la rimozione
				for i := range sameLevel {
					if i >= startIdx {
						sameLevel[i] -= (GroupSize - 1)
					}
				}
			}

			// Se non ci sono più gruppi completi da formare, passa al livello successivo
			if len(sameLevel) < GroupSize {
				break
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
	totalGames := compressed.GamesCount + newRecord.GamesCount

	// Calcola le nuove statistiche
	// Punteggio medio compresso considerando il peso corretto dei record
	newCompressedAvgScore := (compressed.CompressedAverageScore*float64(compressed.GamesCount) +
		newRecord.CompressedAverageScore*float64(newRecord.GamesCount)) / float64(totalGames)

	// Per il record compresso, usa i valori esistenti come base per max/min
	newMaxScore := compressed.MaxScore
	newMinScore := compressed.MinScore

	// Confronta con il nuovo record per aggiornare max/min assoluti
	if newRecord.GamesCount == 1 {
		// Se il nuovo record è una partita singola
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

	// Calcola la nuova media di durata ponderata correttamente
	newCompressedAvgDuration := (compressed.CompressedAverageDuration*float64(compressed.GamesCount) +
		newDuration*float64(newRecord.GamesCount)) / float64(totalGames)

	// Aggiorna max/min duration
	newMaxDuration := compressed.MaxDuration
	newMinDuration := compressed.MinDuration

	if newRecord.GamesCount == 1 {
		// Per record singoli, usa la durata calcolata
		if newDuration > newMaxDuration {
			newMaxDuration = newDuration
		}
		if newDuration < newMinDuration {
			newMinDuration = newDuration
		}
	} else {
		// Per record compressi, usa i valori max/min esistenti
		if newRecord.MaxDuration > newMaxDuration {
			newMaxDuration = newRecord.MaxDuration
		}
		if newRecord.MinDuration < newMinDuration {
			newMinDuration = newRecord.MinDuration
		}
	}

	// Calcola correttamente le medie dei massimi e minimi considerando il peso degli elementi
	var newAverageMaxScore, newAverageMinScore float64
	var newAverageMaxDuration, newAverageMinDuration float64

	if newRecord.GamesCount == 1 {
		// Se è un record singolo, usa il valore del record stesso per le medie
		newAverageMaxScore = (compressed.AverageMaxScore*float64(compressed.GamesCount) +
			float64(newRecord.Score)) / float64(totalGames)

		newAverageMinScore = (compressed.AverageMinScore*float64(compressed.GamesCount) +
			float64(newRecord.Score)) / float64(totalGames)

		newAverageMaxDuration = (compressed.AverageMaxDuration*float64(compressed.GamesCount) +
			newDuration) / float64(totalGames)

		newAverageMinDuration = (compressed.AverageMinDuration*float64(compressed.GamesCount) +
			newDuration) / float64(totalGames)
	} else {
		// Se è un record già compresso, usa le sue medie pesate per il numero di giochi
		newAverageMaxScore = (compressed.AverageMaxScore*float64(compressed.GamesCount) +
			newRecord.AverageMaxScore*float64(newRecord.GamesCount)) / float64(totalGames)

		newAverageMinScore = (compressed.AverageMinScore*float64(compressed.GamesCount) +
			newRecord.AverageMinScore*float64(newRecord.GamesCount)) / float64(totalGames)

		newAverageMaxDuration = (compressed.AverageMaxDuration*float64(compressed.GamesCount) +
			newRecord.AverageMaxDuration*float64(newRecord.GamesCount)) / float64(totalGames)

		newAverageMinDuration = (compressed.AverageMinDuration*float64(compressed.GamesCount) +
			newRecord.AverageMinDuration*float64(newRecord.GamesCount)) / float64(totalGames)
	}

	// Calcola la media dell'entropia della policy ponderata correttamente
	newPolicyEntropy := (compressed.PolicyEntropy*float64(compressed.GamesCount) +
		newRecord.PolicyEntropy*float64(newRecord.GamesCount)) / float64(totalGames)

	// Aggiorna il record compresso con i nuovi valori calcolati
	compressed.GamesCount = totalGames
	compressed.CompressedAverageScore = newCompressedAvgScore
	compressed.MaxScore = newMaxScore
	compressed.MinScore = newMinScore
	compressed.CompressedAverageDuration = newCompressedAvgDuration
	compressed.MaxDuration = newMaxDuration
	compressed.MinDuration = newMinDuration
	compressed.AverageMaxScore = newAverageMaxScore
	compressed.AverageMinScore = newAverageMinScore
	compressed.AverageMaxDuration = newAverageMaxDuration
	compressed.AverageMinDuration = newAverageMinDuration
	compressed.PolicyEntropy = newPolicyEntropy

	// Aggiorna la data di fine se il nuovo record è più recente
	if newRecord.EndTime.After(compressed.EndTime) {
		compressed.EndTime = newRecord.EndTime
	}

	// Rimuovi il record assorbito e aggiorna il record compresso
	newGames := make([]GameRecord, 0, len(s.Games)-1)
	newGames = append(newGames, s.Games[:compressedIdx]...)
	newGames = append(newGames, compressed)
	newGames = append(newGames, s.Games[compressedIdx+1:newIdx]...)
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
