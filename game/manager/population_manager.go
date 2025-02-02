package manager

import (
	"math/rand"
	"snake-game/ai"
	"snake-game/game/entity"
	"snake-game/game/types"
)

const (
	PopGrowthWindow = 100
)

type PopulationManager struct {
	grid           types.Grid
	snakes         []*entity.Snake
	populationHist []int
	reproManager   *ReproductionManager
	currentTick    int
}

func NewPopulationManager(grid types.Grid) *PopulationManager {
	return &PopulationManager{
		grid:           grid,
		snakes:         make([]*entity.Snake, 0, types.MaxPopulation),
		populationHist: make([]int, 0, PopGrowthWindow),
		reproManager:   NewReproductionManager(grid),
		currentTick:    0,
	}
}

func (pm *PopulationManager) InitializePopulation() {
	for i := 0; i < types.NumAgents; i++ {
		// Place agent at random position
		pos := types.Point{
			X: rand.Intn(pm.grid.Width),
			Y: rand.Intn(pm.grid.Height),
		}

		agent := ai.NewQLearning(nil, 0)
		snake := entity.NewSnake(
			pos,
			agent,
			generateRandomColor(),
		)
		pm.snakes = append(pm.snakes, snake)
	}
}

func (pm *PopulationManager) HandleReproduction(snake1, snake2 *entity.Snake) bool {
	if len(pm.snakes) >= types.MaxPopulation {
		return false
	}

	if offspring := pm.reproManager.HandleReproduction(snake1, snake2, pm.currentTick); offspring != nil {
		pm.snakes = append(pm.snakes, offspring)
		return true
	}
	return false
}

func (pm *PopulationManager) Update() {
	pm.currentTick++

	// Update snake ages
	for _, snake := range pm.snakes {
		if !snake.Dead {
			snake.Age++
		}
	}
}

func (pm *PopulationManager) UpdatePopulationHistory() {
	currentPop := len(pm.snakes)
	pm.populationHist = append(pm.populationHist, currentPop)
	if len(pm.populationHist) > PopGrowthWindow {
		pm.populationHist = pm.populationHist[1:]
	}
}

func (pm *PopulationManager) GetSnakes() []*entity.Snake {
	return pm.snakes
}

func (pm *PopulationManager) IsAllSnakesDead() bool {
	for _, snake := range pm.snakes {
		if !snake.Dead {
			return false
		}
	}
	return true
}

func (pm *PopulationManager) GetBestSnakes(n int) []*entity.Snake {
	// Create a copy of snakes slice to sort
	snakesCopy := make([]*entity.Snake, len(pm.snakes))
	copy(snakesCopy, pm.snakes)

	// Sort by score in descending order
	for i := 0; i < len(snakesCopy)-1; i++ {
		for j := i + 1; j < len(snakesCopy); j++ {
			if snakesCopy[i].Score < snakesCopy[j].Score {
				snakesCopy[i], snakesCopy[j] = snakesCopy[j], snakesCopy[i]
			}
		}
	}

	// Return top N snakes or all if less than N
	if len(snakesCopy) < n {
		return snakesCopy
	}
	return snakesCopy[:n]
}

func (pm *PopulationManager) RemoveDeadSnakes() {
	aliveSnakes := make([]*entity.Snake, 0)
	for _, snake := range pm.snakes {
		if !snake.Dead {
			aliveSnakes = append(aliveSnakes, snake)
		} else {
			pm.reproManager.Cleanup(snake)
		}
	}
	pm.snakes = aliveSnakes
}

func (pm *PopulationManager) AddSnake(snake *entity.Snake) {
	pm.snakes = append(pm.snakes, snake)
}

func generateRandomColor() entity.Color {
	return entity.Color{
		R: uint8(rand.Intn(256)),
		G: uint8(rand.Intn(256)),
		B: uint8(rand.Intn(256)),
	}
}
