package main

// Direction constants
const (
	UP    = 0
	RIGHT = 1
	DOWN  = 2
	LEFT  = 3
)

// GetDangers returns whether there are dangers (wall or body) ahead, right, and left
func (g *Game) GetDangers() (ahead, right, left bool) {
	snake := g.snake
	head := snake.GetHead()
	dir := snake.Direction

	// Check position ahead
	aheadPos := Point{
		X: (head.X + dir.X + g.Grid.Width) % g.Grid.Width,
		Y: (head.Y + dir.Y + g.Grid.Height) % g.Grid.Height,
	}
	ahead = g.checkCollision(aheadPos) != NoCollision

	// Calculate right position based on current direction
	rightDir := Point{X: -dir.Y, Y: dir.X} // Rotate 90° clockwise
	rightPos := Point{
		X: (head.X + rightDir.X + g.Grid.Width) % g.Grid.Width,
		Y: (head.Y + rightDir.Y + g.Grid.Height) % g.Grid.Height,
	}
	right = g.checkCollision(rightPos) != NoCollision

	// Calculate left position based on current direction
	leftDir := Point{X: dir.Y, Y: -dir.X} // Rotate 90° counterclockwise
	leftPos := Point{
		X: (head.X + leftDir.X + g.Grid.Width) % g.Grid.Width,
		Y: (head.Y + leftDir.Y + g.Grid.Height) % g.Grid.Height,
	}
	left = g.checkCollision(leftPos) != NoCollision

	return ahead, right, left
}

// GetFoodDirection returns the relative direction of food from the snake's head
func (g *Game) GetFoodDirection() (up, right, down, left bool) {
	head := g.snake.GetHead()
	food := g.food

	// Calculate relative position considering grid wrapping
	dx := food.X - head.X
	if dx > g.Grid.Width/2 {
		dx = dx - g.Grid.Width
	} else if dx < -g.Grid.Width/2 {
		dx = dx + g.Grid.Width
	}

	dy := food.Y - head.Y
	if dy > g.Grid.Height/2 {
		dy = dy - g.Grid.Height
	} else if dy < -g.Grid.Height/2 {
		dy = dy + g.Grid.Height
	}

	// Set direction flags
	up = dy < 0
	down = dy > 0
	right = dx > 0
	left = dx < 0

	return up, right, down, left
}

// GetCurrentDirection returns the current direction of the snake as an integer (0-3)
func (g *Game) GetCurrentDirection() int {
	dir := g.snake.Direction
	switch {
	case dir.Y < 0:
		return UP
	case dir.X > 0:
		return RIGHT
	case dir.Y > 0:
		return DOWN
	case dir.X < 0:
		return LEFT
	default:
		return RIGHT // Default case, should never happen
	}
}
