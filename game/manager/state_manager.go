package manager

import (
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"snake-game/game/entity"
	"snake-game/game/types"
	"time"

	"github.com/google/uuid"
)

type GameStats struct {
	UUID           string       `json:"uuid"`
	PreviousGameID string       `json:"previous_game_id"`
	StartTime      time.Time    `json:"start_time"`
	EndTime        time.Time    `json:"end_time"`
	GameDuration   float64      `json:"game_duration"`
	Iterations     int          `json:"iterations"`
	AgentStats     []AgentStats `json:"agent_stats"`
}

type AgentStats struct {
	UUID         string  `json:"uuid"`
	Score        int     `json:"score"`
	AverageScore float64 `json:"average_score"`
	TotalReward  float64 `json:"total_reward"`
	GamesPlayed  int     `json:"games_played"`
}

type StateManager struct {
	UUID           string
	PreviousGameID string
	grid           types.Grid
	stats          GameStats
	foodList       []types.Point
	foodSpawnTimer int
	iterations     int
	startTime      time.Time
	collisionMgr   *CollisionManager
	popManager     *PopulationManager
}

func NewStateManager(grid types.Grid, previousGameID string, collisionMgr *CollisionManager, popManager *PopulationManager) *StateManager {
	gameUUID := uuid.New().String()
	startTime := time.Now()

	sm := &StateManager{
		UUID:           gameUUID,
		PreviousGameID: previousGameID,
		grid:           grid,
		stats: GameStats{
			UUID:           gameUUID,
			PreviousGameID: previousGameID,
			StartTime:      startTime,
			AgentStats:     make([]AgentStats, 0),
		},
		foodList:       make([]types.Point, 0),
		foodSpawnTimer: 0,
		iterations:     0,
		startTime:      startTime,
		collisionMgr:   collisionMgr,
		popManager:     popManager,
	}

	// Create game directory
	os.MkdirAll(filepath.Join("data", "games", gameUUID, "agents"), 0755)

	return sm
}

func (sm *StateManager) Update(snakes []*entity.Snake) {
	sm.iterations++
	sm.foodSpawnTimer++

	// Update food spawning
	sm.updateFoodSpawning(snakes)

	// Update stats for each snake
	for _, snake := range snakes {
		if snake == nil || snake.Dead {
			continue
		}

		// Update agent stats
		found := false
		for i, stats := range sm.stats.AgentStats {
			if stats.UUID == snake.AI.UUID {
				sm.stats.AgentStats[i].Score = snake.Score
				sm.stats.AgentStats[i].AverageScore = snake.AverageScore
				sm.stats.AgentStats[i].TotalReward = snake.AI.TotalReward
				sm.stats.AgentStats[i].GamesPlayed = snake.AI.GamesPlayed
				found = true
				break
			}
		}

		if !found {
			sm.stats.AgentStats = append(sm.stats.AgentStats, AgentStats{
				UUID:         snake.AI.UUID,
				Score:        snake.Score,
				AverageScore: snake.AverageScore,
				TotalReward:  snake.AI.TotalReward,
				GamesPlayed:  snake.AI.GamesPlayed,
			})
		}
	}
}

func (sm *StateManager) updateFoodSpawning(snakes []*entity.Snake) {
	activeSnakes := 0
	for _, snake := range snakes {
		if snake != nil && !snake.Dead {
			activeSnakes++
		}
	}

	// Always ensure at least one food if there are active snakes
	if activeSnakes > 0 && len(sm.foodList) == 0 {
		newFood := sm.GenerateFood(snakes)
		sm.foodList = append(sm.foodList, newFood)
		sm.foodSpawnTimer = 0
		return
	}

	// Generate new food only if we have less than minimum ratio of food to snakes
	minFood := int(float64(activeSnakes) * types.MinFoodRatio)
	if sm.foodSpawnTimer >= types.FoodSpawnCycles && len(sm.foodList) < minFood {
		newFood := sm.GenerateFood(snakes)
		sm.foodList = append(sm.foodList, newFood)
		sm.foodSpawnTimer = 0
	}
}

func (sm *StateManager) GenerateFood(snakes []*entity.Snake) types.Point {
	for {
		food := types.Point{
			X: rand.Intn(sm.grid.Width),
			Y: rand.Intn(sm.grid.Height),
		}

		if sm.collisionMgr.ValidateSpawnPosition(food, snakes) {
			return food
		}
	}
}

func (sm *StateManager) SaveGameStats() {
	sm.stats.EndTime = time.Now()
	sm.stats.GameDuration = sm.stats.EndTime.Sub(sm.stats.StartTime).Seconds()
	sm.stats.Iterations = sm.iterations

	// Save game stats to JSON file
	statsPath := filepath.Join("data", "games", sm.UUID, "game_stats.json")
	statsJson, _ := json.Marshal(sm.stats)
	os.WriteFile(statsPath, statsJson, 0644)
}

func (sm *StateManager) GetIterations() int {
	return sm.iterations
}

func (sm *StateManager) GetFoodList() []types.Point {
	return sm.foodList
}

func (sm *StateManager) AddFood(food types.Point) {
	sm.foodList = append(sm.foodList, food)
}

func (sm *StateManager) RemoveFood(food types.Point) {
	for i, f := range sm.foodList {
		if f == food {
			// Remove food from list by swapping with last element and truncating
			sm.foodList[i] = sm.foodList[len(sm.foodList)-1]
			sm.foodList = sm.foodList[:len(sm.foodList)-1]
			return
		}
	}
}

func (sm *StateManager) GetStats() GameStats {
	return sm.stats
}

func (sm *StateManager) GetPopulationManager() *PopulationManager {
	return sm.popManager
}
