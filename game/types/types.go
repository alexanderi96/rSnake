package types

// Grid represents the game grid dimensions
type Grid struct {
	Width  int
	Height int
}

// Game constants
const (
	FoodSpawnCycles = 30  // Cycles between food spawns
	MinFoodRatio    = 1.0 // Ensure at least one food per snake
)
