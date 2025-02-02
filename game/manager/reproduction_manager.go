package manager

import (
	"math/rand"
	"snake-game/ai"
	"snake-game/game/entity"
	"snake-game/game/types"
)

const (
	MinReproductionAge   = 100 // Minimum ticks before a snake can reproduce
	ReproductionCooldown = 200 // Ticks between reproduction attempts
	MaxOffspringDistance = 3   // Maximum distance for offspring spawn
	MinOffspringDistance = 1   // Minimum distance for offspring spawn
	MaxReproductionTries = 10  // Maximum attempts to find valid spawn position
)

type ReproductionManager struct {
	grid                 types.Grid
	lastReproductionTime map[*entity.Snake]int
}

func NewReproductionManager(grid types.Grid) *ReproductionManager {
	return &ReproductionManager{
		grid:                 grid,
		lastReproductionTime: make(map[*entity.Snake]int),
	}
}

func (rm *ReproductionManager) CanReproduce(snake *entity.Snake, currentTick int) bool {
	// Check if snake is mature enough
	if snake.Age < MinReproductionAge {
		return false
	}

	// Check cooldown
	lastRepro, exists := rm.lastReproductionTime[snake]
	if exists && currentTick-lastRepro < ReproductionCooldown {
		return false
	}

	return true
}

func (rm *ReproductionManager) HandleReproduction(snake1, snake2 *entity.Snake, currentTick int) *entity.Snake {
	// Verify both snakes can reproduce
	if !rm.CanReproduce(snake1, currentTick) || !rm.CanReproduce(snake2, currentTick) {
		return nil
	}

	head1 := snake1.GetHead()
	head2 := snake2.GetHead()

	// Check if snakes are exactly one cube apart
	dx := head2.X - head1.X
	dy := head2.Y - head1.Y
	if abs(dx)+abs(dy) != 1 { // Manhattan distance must be 1
		return nil
	}

	// Check if snakes are facing each other
	if !areFacingEachOther(snake1.Direction, snake2.Direction) {
		return nil
	}

	// Spawn position is between the two snakes
	spawnPos := types.Point{
		X: head1.X + dx/2,
		Y: head1.Y + dy/2,
	}

	// Create offspring at the midpoint
	offspring := rm.createOffspring(snake1, snake2, spawnPos)

	// Update reproduction timestamps
	rm.lastReproductionTime[snake1] = currentTick
	rm.lastReproductionTime[snake2] = currentTick

	return offspring
}

func (rm *ReproductionManager) createOffspring(parent1, parent2 *entity.Snake, position types.Point) *entity.Snake {
	// Create new AI through breeding
	newAgent := ai.Breed(parent1.AI, parent2.AI)

	// Create new snake with mixed traits
	newSnake := entity.NewSnake(
		position,
		newAgent,
		mutateColor(parent1.Color),
	)

	// Initialize offspring properties
	newSnake.Age = 0

	return newSnake
}

func (rm *ReproductionManager) Cleanup(snake *entity.Snake) {
	delete(rm.lastReproductionTime, snake)
}

// Helper function to check if snakes are facing each other
func areFacingEachOther(dir1, dir2 types.Point) bool {
	return dir1.X == -dir2.X && dir1.Y == -dir2.Y
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func mutateColor(baseColor entity.Color) entity.Color {
	mutateComponent := func(c uint8) uint8 {
		delta := uint8(rand.Intn(61) - 30) // Range: -30 to +30
		if delta > c {
			return delta - c // Handle underflow
		}
		if uint16(c)+uint16(delta) > 255 {
			return 255 // Handle overflow
		}
		return c + delta
	}

	return entity.Color{
		R: mutateComponent(baseColor.R),
		G: mutateComponent(baseColor.G),
		B: mutateComponent(baseColor.B),
	}
}
