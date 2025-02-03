package manager

import (
	"encoding/json"
	"os"
	"snake-game/game/entity"
	"snake-game/game/types"
)

type GameStats struct {
	HighScore    int   `json:"highScore"`
	ScoreHistory []int `json:"scoreHistory"`
}

type StateManager struct {
	grid         types.Grid
	collisionMgr *CollisionManager
	popManager   *PopulationManager
	foodManager  *FoodManager
	highScore    int
	scoreHistory []int
}

func NewStateManager(grid types.Grid, collisionMgr *CollisionManager, popManager *PopulationManager) *StateManager {
	sm := &StateManager{
		grid:         grid,
		collisionMgr: collisionMgr,
		popManager:   popManager,
		foodManager:  NewFoodManager(grid, collisionMgr),
		highScore:    0,
		scoreHistory: make([]int, 0),
	}

	// Create data directory if it doesn't exist
	if err := os.MkdirAll("data", 0755); err != nil {
		println("Warning: Could not create data directory:", err)
	}

	// Load saved stats
	if err := sm.LoadStats("data/gamestats.json"); err == nil {
		// Stats loaded successfully
	}

	return sm
}

func (sm *StateManager) SaveStats(filename string) error {
	stats := GameStats{
		HighScore:    sm.highScore,
		ScoreHistory: sm.scoreHistory,
	}

	data, err := json.MarshalIndent(stats, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filename, data, 0644)
}

func (sm *StateManager) LoadStats(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	var stats GameStats
	if err := json.Unmarshal(data, &stats); err != nil {
		return err
	}

	sm.highScore = stats.HighScore
	sm.scoreHistory = stats.ScoreHistory
	return nil
}

func (sm *StateManager) Update(snakes []*entity.Snake) {
	// Update food manager
	sm.foodManager.Update(snakes)
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

func (sm *StateManager) GetPopulationManager() *PopulationManager {
	return sm.popManager
}

func (sm *StateManager) UpdateScore(score int) {
	if score > sm.highScore {
		sm.highScore = score
	}
	sm.SaveStats("data/gamestats.json")
}

func (sm *StateManager) AddToHistory(score int) {
	sm.scoreHistory = append(sm.scoreHistory, score)
	sm.SaveStats("data/gamestats.json")
}

func (sm *StateManager) GetHighScore() int {
	return sm.highScore
}

func (sm *StateManager) GetScoreHistory() []int {
	return sm.scoreHistory
}
