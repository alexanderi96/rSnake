package manager

import (
	"math/rand"
	"snake-game/game/entity"
	"snake-game/game/types"
)

type FoodManager struct {
	grid         types.Grid
	foodList     []types.Point
	spawnTimer   int
	collisionMgr *CollisionManager
}

func NewFoodManager(grid types.Grid, collisionMgr *CollisionManager) *FoodManager {
	return &FoodManager{
		grid:         grid,
		foodList:     make([]types.Point, 0),
		spawnTimer:   0,
		collisionMgr: collisionMgr,
	}
}

func (fm *FoodManager) Update(snakes []*entity.Snake) {
	fm.spawnTimer++
	fm.updateFoodSpawning(snakes)
}

func (fm *FoodManager) updateFoodSpawning(snakes []*entity.Snake) {
	activeSnakes := 0
	for _, snake := range snakes {
		if snake != nil && !snake.Dead {
			activeSnakes++
		}
	}

	// Always ensure at least one food if there are active snakes
	if activeSnakes > 0 && len(fm.foodList) == 0 {
		newFood := fm.GenerateFood(snakes)
		fm.foodList = append(fm.foodList, newFood)
		fm.spawnTimer = 0
		return
	}

	// Generate new food only if we have less than minimum ratio of food to snakes
	minFood := int(float64(activeSnakes) * types.MinFoodRatio)
	if fm.spawnTimer >= types.FoodSpawnCycles && len(fm.foodList) < minFood {
		newFood := fm.GenerateFood(snakes)
		fm.foodList = append(fm.foodList, newFood)
		fm.spawnTimer = 0
	}
}

func (fm *FoodManager) GenerateFood(snakes []*entity.Snake) types.Point {
	for {
		food := types.Point{
			X: rand.Intn(fm.grid.Width),
			Y: rand.Intn(fm.grid.Height),
		}

		if fm.collisionMgr.ValidateSpawnPosition(food, snakes) {
			return food
		}
	}
}

func (fm *FoodManager) GetFoodList() []types.Point {
	return fm.foodList
}

func (fm *FoodManager) AddFood(food types.Point) {
	fm.foodList = append(fm.foodList, food)
}

func (fm *FoodManager) RemoveFood(food types.Point) {
	for i, f := range fm.foodList {
		if f == food {
			// Remove food from list by swapping with last element and truncating
			fm.foodList[i] = fm.foodList[len(fm.foodList)-1]
			fm.foodList = fm.foodList[:len(fm.foodList)-1]
			return
		}
	}
}
