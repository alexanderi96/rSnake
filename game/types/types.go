package types

// Grid represents the game grid dimensions
type Grid struct {
	Width  int
	Height int
}

// Game constants
const (
	NumAgents       = 4   // Number of initial agents
	MaxPopulation   = 12  // Maximum allowed population
	FoodSpawnCycles = 30  // Reduced from 100 to spawn food more frequently
	MinFoodRatio    = 1.0 // Ensure at least one food per snake
)
