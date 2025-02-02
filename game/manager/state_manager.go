package manager

import (
	"encoding/json"
	"os"
	"path/filepath"
	"snake-game/game/entity"
	"snake-game/game/types"
	"time"

	"github.com/google/uuid"
)

type GameStats struct {
	UUID             string       `json:"uuid"`
	PreviousGameID   string       `json:"previous_game_id"`
	SessionStartTime time.Time    `json:"session_start_time"` // When the training session started
	GameStartTime    time.Time    `json:"game_start_time"`    // When this specific game round started
	GameEndTime      time.Time    `json:"game_end_time"`      // When this game round ended
	GameDuration     float64      `json:"game_duration"`      // Duration of this specific game round
	TotalSessionTime float64      `json:"total_session_time"` // Total time since session start
	RoundsPlayed     int          `json:"rounds_played"`      // Number of game rounds played in this session
	CurrentRound     int          `json:"current_round"`      // Current game round number
	Iterations       int          `json:"iterations"`
	AgentStats       []AgentStats `json:"agent_stats"`
}

type AgentStats struct {
	UUID         string  `json:"uuid"`
	Score        int     `json:"score"`
	AverageScore float64 `json:"average_score"`
	TotalReward  float64 `json:"total_reward"`
	RoundsPlayed int     `json:"rounds_played"` // Number of rounds this agent has played
}

type StateManager struct {
	UUID           string
	PreviousGameID string
	grid           types.Grid
	stats          GameStats
	iterations     int
	sessionStart   time.Time // When the training session started
	gameStartTime  time.Time // When the current game round started
	roundsPlayed   int       // Total number of game rounds played
	collisionMgr   *CollisionManager
	popManager     *PopulationManager
	foodManager    *FoodManager
}

func NewStateManager(grid types.Grid, previousGameID string, collisionMgr *CollisionManager, popManager *PopulationManager) *StateManager {
	gameUUID := uuid.New().String()
	now := time.Now()

	// Ensure data directories exist
	os.MkdirAll(filepath.Join("data", "games"), 0755)
	os.MkdirAll(filepath.Join("data", "games", gameUUID, "agents"), 0755)

	sm := &StateManager{
		UUID:           gameUUID,
		PreviousGameID: previousGameID,
		grid:           grid,
		stats: GameStats{
			UUID:             gameUUID,
			PreviousGameID:   previousGameID,
			SessionStartTime: now,
			GameStartTime:    now,
			RoundsPlayed:     0,
			CurrentRound:     1,
			AgentStats:       make([]AgentStats, 0),
		},
		iterations:    0,
		sessionStart:  now,
		gameStartTime: now,
		roundsPlayed:  0,
		collisionMgr:  collisionMgr,
		popManager:    popManager,
		foodManager:   NewFoodManager(grid, collisionMgr),
	}

	return sm
}

func (sm *StateManager) Update(snakes []*entity.Snake) {
	sm.iterations++

	// Check if all snakes are dead to reset game start time
	allDead := true
	for _, snake := range snakes {
		if snake == nil {
			continue
		}
		snake.Mutex.RLock()
		isDead := snake.Dead
		snake.Mutex.RUnlock()
		if !isDead {
			allDead = false
			break
		}
	}
	if allDead {
		sm.gameStartTime = time.Now()
	}

	// Update food manager
	sm.foodManager.Update(snakes)

	// Update stats for each snake
	for _, snake := range snakes {
		if snake == nil {
			continue
		}

		snake.Mutex.RLock()
		isDead := snake.Dead
		if isDead {
			snake.Mutex.RUnlock()
			continue
		}

		// Copy values while holding lock
		score := snake.Score
		avgScore := snake.AverageScore
		aiUUID := snake.AI.UUID
		totalReward := snake.AI.TotalReward
		gamesPlayed := snake.AI.GamesPlayed
		snake.Mutex.RUnlock()

		// Update agent stats
		found := false
		for i, stats := range sm.stats.AgentStats {
			if stats.UUID == aiUUID {
				sm.stats.AgentStats[i].Score = score
				sm.stats.AgentStats[i].AverageScore = avgScore
				sm.stats.AgentStats[i].TotalReward = totalReward
				sm.stats.AgentStats[i].RoundsPlayed = gamesPlayed
				found = true
				break
			}
		}

		if !found {
			sm.stats.AgentStats = append(sm.stats.AgentStats, AgentStats{
				UUID:         aiUUID,
				Score:        score,
				AverageScore: avgScore,
				TotalReward:  totalReward,
				RoundsPlayed: gamesPlayed,
			})
		}
	}
}

func (sm *StateManager) GetFoodList() []types.Point {
	return sm.foodManager.GetFoodList()
}

func (sm *StateManager) AddFood(food types.Point) {
	sm.foodManager.AddFood(food)
}

func (sm *StateManager) RemoveFood(food types.Point) {
	sm.foodManager.RemoveFood(food)
}

func (sm *StateManager) GenerateFood(snakes []*entity.Snake) types.Point {
	return sm.foodManager.GenerateFood(snakes)
}

type GlobalStats struct {
	TrainingStartTime time.Time `json:"training_start_time"`
	TotalGamesStarted int       `json:"total_games_started"`
	LastGameID        string    `json:"last_game_id"`
	LastUpdateTime    time.Time `json:"last_update_time"`
}

func (sm *StateManager) SaveGameStats() {
	now := time.Now()

	// Update game round stats
	sm.stats.GameEndTime = now
	sm.stats.GameDuration = now.Sub(sm.gameStartTime).Seconds()
	sm.stats.TotalSessionTime = now.Sub(sm.sessionStart).Seconds()
	sm.stats.Iterations = sm.iterations
	sm.stats.RoundsPlayed++
	sm.stats.CurrentRound++

	// Save game stats to JSON file
	statsPath := filepath.Join("data", "games", sm.UUID, "game_stats.json")
	statsJson, _ := json.Marshal(sm.stats)
	os.WriteFile(statsPath, statsJson, 0644)

	// Update global stats
	globalStatsPath := filepath.Join("data", "global_stats.json")
	var globalStats GlobalStats

	// Try to read existing global stats
	if data, err := os.ReadFile(globalStatsPath); err == nil {
		json.Unmarshal(data, &globalStats)
	} else {
		// Initialize new global stats if file doesn't exist
		globalStats = GlobalStats{
			TrainingStartTime: sm.sessionStart,
			TotalGamesStarted: 0,
		}
	}

	// Update global stats
	globalStats.TotalGamesStarted++
	globalStats.LastGameID = sm.UUID
	globalStats.LastUpdateTime = now

	// Save updated global stats
	globalStatsJson, _ := json.Marshal(globalStats)
	os.WriteFile(globalStatsPath, globalStatsJson, 0644)
}

func (sm *StateManager) GetStats() (GameStats, time.Time) {
	return sm.stats, sm.gameStartTime
}

func (sm *StateManager) GetPopulationManager() *PopulationManager {
	return sm.popManager
}
