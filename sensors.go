package main

// Direction rappresenta una direzione cardinale
type Direction int

const (
	NONE  Direction = iota // 0
	UP                     // 1
	RIGHT                  // 2
	DOWN                   // 3
	LEFT                   // 4
)

// ToPoint converte una Direction in un vettore di spostamento
func (d Direction) ToPoint() Point {
	switch d {
	case UP:
		return Point{X: 0, Y: -1} // Su (decrementa Y)
	case RIGHT:
		return Point{X: 1, Y: 0} // Destra (incrementa X)
	case DOWN:
		return Point{X: 0, Y: 1} // Giù (incrementa Y)
	case LEFT:
		return Point{X: -1, Y: 0} // Sinistra (decrementa X)
	default:
		return Point{X: 0, Y: 0}
	}
}

// TurnLeft restituisce la direzione risultante da una rotazione a sinistra rispetto alla direzione corrente.
func (d Direction) TurnLeft() Direction {
	switch d {
	case UP:
		return LEFT
	case RIGHT:
		return UP
	case DOWN:
		return RIGHT
	case LEFT:
		return DOWN
	default:
		return d
	}
}

// TurnRight restituisce la direzione risultante da una rotazione a destra rispetto alla direzione corrente.
func (d Direction) TurnRight() Direction {
	switch d {
	case UP:
		return RIGHT
	case RIGHT:
		return DOWN
	case DOWN:
		return LEFT
	case LEFT:
		return UP
	default:
		return d
	}
}

// GetDangers restituisce i pericoli relativi alla direzione corrente dello snake.
// Vengono restituiti tre booleani indicanti rispettivamente:
//   - dangerAhead: pericolo nella direzione "avanti"
//   - dangerLeft: pericolo a sinistra
//   - dangerRight: pericolo a destra
func (g *Game) GetDangers() (dangerAhead, dangerLeft, dangerRight bool) {
	snake := g.snake
	head := snake.GetHead()
	currentDir := g.GetCurrentDirection()

	// Calcola la posizione "avanti" rispetto alla direzione attuale,
	// considerando il wrapping della griglia.
	aheadVector := currentDir.ToPoint()
	aheadPos := Point{
		X: (head.X + aheadVector.X + g.Grid.Width) % g.Grid.Width,
		Y: (head.Y + aheadVector.Y + g.Grid.Height) % g.Grid.Height,
	}
	dangerAhead = g.checkCollision(aheadPos) != NoCollision

	// Calcola la posizione a sinistra (rotazione a sinistra)
	leftVector := currentDir.TurnLeft().ToPoint()
	leftPos := Point{
		X: (head.X + leftVector.X + g.Grid.Width) % g.Grid.Width,
		Y: (head.Y + leftVector.Y + g.Grid.Height) % g.Grid.Height,
	}
	dangerLeft = g.checkCollision(leftPos) != NoCollision

	// Calcola la posizione a destra (rotazione a destra)
	rightVector := currentDir.TurnRight().ToPoint()
	rightPos := Point{
		X: (head.X + rightVector.X + g.Grid.Width) % g.Grid.Width,
		Y: (head.Y + rightVector.Y + g.Grid.Height) % g.Grid.Height,
	}
	dangerRight = g.checkCollision(rightPos) != NoCollision

	return dangerAhead, dangerLeft, dangerRight
}

// GetFoodDirection restituisce le flag booleane per le direzioni assolute in cui si trova il cibo,
// rispetto alla testa dello snake.
func (g *Game) GetFoodDirection() (up, right, down, left bool) {
	head := g.snake.GetHead()
	food := g.food

	// Calcola la differenza considerando il wrapping della griglia
	dx := food.X - head.X
	if dx > g.Grid.Width/2 {
		dx -= g.Grid.Width
	} else if dx < -g.Grid.Width/2 {
		dx += g.Grid.Width
	}

	dy := food.Y - head.Y
	if dy > g.Grid.Height/2 {
		dy -= g.Grid.Height
	} else if dy < -g.Grid.Height/2 {
		dy += g.Grid.Height
	}

	up = dy < 0
	down = dy > 0
	right = dx > 0
	left = dx < 0

	return up, right, down, left
}

// GetCurrentDirection restituisce la direzione assoluta corrente dello snake,
// interpretando il vettore velocità (snake.Direction) in una delle direzioni cardinali.
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
		return RIGHT // Caso di fallback
	}
}
