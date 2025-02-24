package main

import (
	"encoding/json"
	"os"
	"snake-game/qlearning"
	"sync"
	"time"
)

const StatsFile = qlearning.DataDir + "/stats.json"

// GameRecord rappresenta i dati di una singola partita.
type GameRecord struct {
	StartTime time.Time `json:"startTime"`
	EndTime   time.Time `json:"endTime"`
	Score     int       `json:"score"`
}

// GameStatsData è il formato usato per salvare/leggere i dati da file.
type GameStatsData struct {
	Games []GameRecord `json:"games"`
}

// GameStats contiene tutte le partite registrate e fornisce metodi per
// ottenere statistiche come punteggio medio, durata media, ecc.
type GameStats struct {
	Games []GameRecord // Array di record di partita
	mutex sync.RWMutex
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
		StartTime: startTime,
		EndTime:   endTime,
		Score:     score,
	}
	s.Games = append(s.Games, game)
	// Se vuoi salvare subito su file, puoi decommentare la riga seguente:
	// s.saveToFile()
}

// GetStats restituisce i dati attuali delle statistiche.
func (s *GameStats) GetStats() GameStatsData {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	return GameStatsData{
		Games: s.Games,
	}
}

// GetAverageScore calcola e restituisce il punteggio medio.
func (s *GameStats) GetAverageScore() float64 {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if len(s.Games) == 0 {
		return 0
	}

	total := 0
	for _, game := range s.Games {
		total += game.Score
	}
	return float64(total) / float64(len(s.Games))
}

// GetMaxScore restituisce il punteggio massimo registrato.
func (s *GameStats) GetMaxScore() int {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if len(s.Games) == 0 {
		return 0
	}

	maxScore := s.Games[0].Score
	for _, game := range s.Games {
		if game.Score > maxScore {
			maxScore = game.Score
		}
	}
	return maxScore
}

// GetGamesPlayed restituisce il numero di partite giocate.
func (s *GameStats) GetGamesPlayed() int {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	return len(s.Games)
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

// GetMaxDuration restituisce la durata massima di una partita (in secondi).
func (s *GameStats) GetMaxDuration() float64 {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	if len(s.Games) == 0 {
		return 0
	}

	maxDuration := s.Games[0].EndTime.Sub(s.Games[0].StartTime).Seconds()
	for _, game := range s.Games {
		duration := game.EndTime.Sub(game.StartTime).Seconds()
		if duration > maxDuration {
			maxDuration = duration
		}
	}
	return maxDuration
}

// saveToFile salva le statistiche su file in formato JSON.
func (s *GameStats) saveToFile() error {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	data := GameStatsData{
		Games: s.Games,
	}

	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}

	return os.WriteFile(StatsFile, jsonData, 0644)
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

	var statsData GameStatsData
	if err := json.Unmarshal(data, &statsData); err != nil {
		return err
	}

	s.Games = statsData.Games
	return nil
}
