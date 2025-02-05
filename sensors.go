package main

// Direction represents a cardinal direction
type Direction int

const (
	NONE  Direction = iota //0
	UP                     // 1
	RIGHT                  // 2
	DOWN                   // 3
	LEFT                   // 4
)

// ToPoint converts a Direction to its corresponding movement vector
func (d Direction) ToPoint() Point {
	switch d {
	case UP:
		return Point{X: 0, Y: -1} // Move up (decrease Y)
	case RIGHT:
		return Point{X: 1, Y: 0} // Move right (increase X)
	case DOWN:
		return Point{X: 0, Y: 1} // Move down (increase Y)
	case LEFT:
		return Point{X: -1, Y: 0} // Move left (decrease X)
	default:
		return Point{X: 0, Y: 0}
	}
}

func (g *Game) GetDangers() (dangerUp, dangerDown, dangerLeft, dangerRight bool) {
	snake := g.snake
	head := snake.GetHead()

	// Danger Up: controllo sulla cella sopra la testa (Y - 1)
	upPos := Point{
		X: head.X,
		Y: (head.Y - 1 + g.Grid.Height) % g.Grid.Height,
	}
	dangerUp = g.checkCollision(upPos) != NoCollision

	// Danger Down: controllo sulla cella sotto la testa (Y + 1)
	downPos := Point{
		X: head.X,
		Y: (head.Y + 1) % g.Grid.Height,
	}
	dangerDown = g.checkCollision(downPos) != NoCollision

	// Danger Left: controllo sulla cella a sinistra della testa (X - 1)
	leftPos := Point{
		X: (head.X - 1 + g.Grid.Width) % g.Grid.Width,
		Y: head.Y,
	}
	dangerLeft = g.checkCollision(leftPos) != NoCollision

	// Danger Right: controllo sulla cella a destra della testa (X + 1)
	rightPos := Point{
		X: (head.X + 1) % g.Grid.Width,
		Y: head.Y,
	}
	dangerRight = g.checkCollision(rightPos) != NoCollision

	return dangerUp, dangerDown, dangerLeft, dangerRight
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

// GetCurrentDirection returns the current direction of the snake
func (g *Game) GetCurrentDirection() Direction {
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
