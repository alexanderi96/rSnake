package main

import (
	"encoding/json"
	"os"
	"sync"
	"time"
)

type GameRecord struct {
	StartTime time.Time `json:"startTime"`
	EndTime   time.Time `json:"endTime"`
	Score     int       `json:"score"`
}

type GameStats struct {
	Games []GameRecord // Array of individual game records
	mutex sync.RWMutex
}

type GameStatsData struct {
	Games []GameRecord `json:"games"`
}

func NewGameStats() *GameStats {
	stats := &GameStats{
		Games: make([]GameRecord, 0),
	}
	stats.loadFromFile()
	return stats
}

func (s *GameStats) AddGame(score int, startTime, endTime time.Time) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	game := GameRecord{
		StartTime: startTime,
		EndTime:   endTime,
		Score:     score,
	}
	s.Games = append(s.Games, game)
	s.saveToFile()
}

func (s *GameStats) GetStats() GameStatsData {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	return GameStatsData{
		Games: s.Games,
	}
}

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

func (s *GameStats) GetGamesPlayed() int {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	return len(s.Games)
}

func (s *GameStats) saveToFile() error {
	data := GameStatsData{
		Games: s.Games,
	}

	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}

	return os.WriteFile("data/game_stats.json", jsonData, 0644)
}

func (s *GameStats) backupStats() error {
	// Create backup directory if it doesn't exist
	if err := os.MkdirAll("data/backup", 0755); err != nil {
		return err
	}

	// Read current stats
	data, err := os.ReadFile("data/game_stats.json")
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No stats to backup
		}
		return err
	}

	// Create backup filename with timestamp
	timestamp := time.Now().Format("20060102_150405")
	backupPath := "data/backup/game_stats_" + timestamp + ".json"

	// Write backup file
	return os.WriteFile(backupPath, data, 0644)
}

func (s *GameStats) Reset() error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	// Backup current stats
	if err := s.backupStats(); err != nil {
		return err
	}

	// Reset stats
	s.Games = make([]GameRecord, 0)
	return s.saveToFile()
}

func (s *GameStats) loadFromFile() error {
	data, err := os.ReadFile("data/game_stats.json")
	if err != nil {
		if os.IsNotExist(err) {
			s.Games = make([]GameRecord, 0)
			return nil // File doesn't exist yet, start with empty stats
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
