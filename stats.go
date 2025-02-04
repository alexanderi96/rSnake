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

	return os.WriteFile("game_stats.json", jsonData, 0644)
}

func (s *GameStats) loadFromFile() error {
	data, err := os.ReadFile("game_stats.json")
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
