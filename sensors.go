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

// GetDangers restituisce le distanze dai pericoli rispetto alla direzione corrente dello snake.
// Vengono restituiti:
//   - distAhead, distLeft, distRight: distanza dal primo pericolo in quella direzione
//     (0 se non c'è pericolo entro 5 celle, 1 se adiacente, >1 per distanze maggiori)
func (g *Game) GetDangers() (distAhead, distLeft, distRight int) {
	snake := g.snake
	head := snake.GetHead()
	currentDir := g.GetCurrentDirection()

	// Funzione helper per calcolare la distanza dal pericolo in una direzione
	getDangerDistance := func(dir Direction) int {
		vector := dir.ToPoint()
		pos := head
		for dist := 1; dist <= 5; dist++ { // Controlliamo fino a 5 celle di distanza
			pos = Point{
				X: (pos.X + vector.X + g.Grid.Width) % g.Grid.Width,
				Y: (pos.Y + vector.Y + g.Grid.Height) % g.Grid.Height,
			}
			if g.checkCollision(pos) != NoCollision {
				return dist
			}
		}
		return 0 // Nessun pericolo entro 5 celle
	}

	// Calcola le distanze in ogni direzione
	distAhead = getDangerDistance(currentDir)
	distLeft = getDangerDistance(currentDir.TurnLeft())
	distRight = getDangerDistance(currentDir.TurnRight())

	return
}

// GetFoodDirection restituisce la direzione assoluta principale in cui si trova il cibo rispetto alla testa dello snake.
func (g *Game) GetFoodDirection() Direction {
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

	// Restituisce la direzione predominante (se il cibo è in diagonale, sceglie l'asse con la distanza maggiore)
	if abs(dx) > abs(dy) {
		if dx > 0 {
			return RIGHT
		}
		return LEFT
	} else {
		if dy > 0 {
			return DOWN
		}
		return UP
	}
}

// GetRelativeFoodDirection restituisce la direzione relativa del cibo rispetto allo snake.
func (g *Game) GetRelativeFoodDirection() Direction {
	absoluteFoodDir := g.GetFoodDirection()
	snakeDir := g.GetCurrentDirection()

	// Converte la direzione assoluta del cibo in relativa rispetto alla direzione dello snake
	return Direction((absoluteFoodDir - snakeDir + 4) % 4)
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
