package manager

import (
	"math/rand"
	"snake-game/ai"
	"snake-game/game/entity"
	"snake-game/game/types"
)

type PopulationManager struct {
	grid         types.Grid
	currentSnake *entity.Snake
	lastSnake    *entity.Snake // Parent for next snake
}

func NewPopulationManager(grid types.Grid) *PopulationManager {
	return &PopulationManager{
		grid:         grid,
		currentSnake: nil,
		lastSnake:    nil,
	}
}

func (pm *PopulationManager) InitializePopulation() {
	if pm.currentSnake == nil {
		// Random starting position
		pos := types.Point{
			X: rand.Intn(pm.grid.Width),
			Y: rand.Intn(pm.grid.Height),
		}

		// Create AI agent
		var agent *ai.QLearning
		if pm.lastSnake != nil {
			// Inherit parent's knowledge with small mutation
			agent = ai.NewQLearning(pm.lastSnake.AI.QTable, 0.01)
		} else {
			// First snake starts with empty knowledge
			agent = ai.NewQLearning(nil, 0)
		}

		// Create new snake
		pm.currentSnake = entity.NewSnake(pos, agent, generateRandomColor())
	}
}

func (pm *PopulationManager) GetSnakes() []*entity.Snake {
	if pm.currentSnake == nil {
		return []*entity.Snake{}
	}
	return []*entity.Snake{pm.currentSnake}
}

func (pm *PopulationManager) Update() {
	// Empty implementation since we don't track age anymore
}

func (pm *PopulationManager) IsAllSnakesDead() bool {
	return pm.currentSnake == nil || pm.currentSnake.Dead
}

func (pm *PopulationManager) RemoveDeadSnakes() {
	if pm.currentSnake != nil && pm.currentSnake.Dead {
		// Save Q-table before removing the snake
		if pm.currentSnake.AI != nil {
			pm.currentSnake.AI.SaveQTable("data/qtable.json")
		}

		pm.lastSnake = pm.currentSnake // Store for inheritance
		pm.currentSnake = nil          // Remove dead snake
		pm.InitializePopulation()      // Create new snake immediately
	}
}

func (pm *PopulationManager) AddSnake(snake *entity.Snake) {
	pm.currentSnake = snake
}

func generateRandomColor() entity.Color {
	return entity.Color{
		R: uint8(rand.Intn(256)),
		G: uint8(rand.Intn(256)),
		B: uint8(rand.Intn(256)),
	}
}
