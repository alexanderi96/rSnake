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

// GetCombinedDirectionalInfo restituisce un valore combinato tra -1 e 1 per una data direzione,
// considerando sia la presenza di cibo che di pericoli, con scaling basato sulla distanza
func (g *Game) GetCombinedDirectionalInfo(dir Point) float64 {
	head := g.snake.GetHead()

	// Calcola la posizione nella direzione specificata
	nextPos := Point{
		X: (head.X + dir.X + g.Grid.Width) % g.Grid.Width,
		Y: (head.Y + dir.Y + g.Grid.Height) % g.Grid.Height,
	}

	// Ottiene informazioni dettagliate sulla collisione
	_, collisionInfo := g.getCollisionInfo(nextPos)

	// Calcola il valore di pericolo basato sulla distanza
	var dangerValue float64
	if collisionInfo.Type != NoCollision {
		if collisionInfo.Distance == 0 {
			return -1.0 // Collisione immediata
		}
		// Scala il pericolo in base alla distanza (più vicino = più pericoloso)
		dangerValue = -1.0 / float64(collisionInfo.Distance)
	}

	// Calcola la distanza dal cibo
	foodDist := g.getManhattanDistance(nextPos, g.food)
	currentDist := g.getManhattanDistance(head, g.food)

	// Se la nuova posizione è il cibo
	if nextPos == g.food {
		return 1.0 // Cibo presente
	}

	// Se ci stiamo avvicinando al cibo
	if foodDist < currentDist {
		foodValue := 0.5 // Base value per direzione favorevole
		// Se non c'è pericolo, mantiene il valore positivo
		if dangerValue == 0 {
			return foodValue
		}
		// Altrimenti combina il valore del cibo con il pericolo
		return maxFloat64(dangerValue, foodValue)
	}

	// Se ci stiamo allontanando dal cibo
	if foodDist > currentDist {
		foodValue := -0.3 // Base value per direzione sfavorevole
		// Combina con il valore di pericolo
		return minFloat64(dangerValue, foodValue)
	}

	// Se non c'è cibo, ritorna solo il valore di pericolo
	return dangerValue
}

// maxFloat64 returns the larger of two float64s
func maxFloat64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

// minFloat64 returns the smaller of two float64s
func minFloat64(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// GetStateInfo restituisce i valori combinati per le 5 direzioni principali
func (g *Game) GetStateInfo() (front, left, right, frontLeft, frontRight float64) {
	currentDir := g.GetCurrentDirection()
	leftDir := currentDir.TurnLeft()
	rightDir := currentDir.TurnRight()

	// Calcola i vettori per le direzioni diagonali
	frontLeftVec := Point{
		X: currentDir.ToPoint().X + leftDir.ToPoint().X,
		Y: currentDir.ToPoint().Y + leftDir.ToPoint().Y,
	}
	frontRightVec := Point{
		X: currentDir.ToPoint().X + rightDir.ToPoint().X,
		Y: currentDir.ToPoint().Y + rightDir.ToPoint().Y,
	}

	// Ottiene i valori combinati per ogni direzione
	front = g.GetCombinedDirectionalInfo(currentDir.ToPoint())
	left = g.GetCombinedDirectionalInfo(leftDir.ToPoint())
	right = g.GetCombinedDirectionalInfo(rightDir.ToPoint())
	frontLeft = g.GetCombinedDirectionalInfo(frontLeftVec)
	frontRight = g.GetCombinedDirectionalInfo(frontRightVec)

	return
}

// getManhattanDistance calcola la distanza Manhattan tra due punti considerando i bordi della griglia
func (g *Game) getManhattanDistance(p1, p2 Point) int {
	dx := abs(p1.X - p2.X)
	dy := abs(p1.Y - p2.Y)

	// Considera il wrapping della griglia
	if dx > g.Grid.Width/2 {
		dx = g.Grid.Width - dx
	}
	if dy > g.Grid.Height/2 {
		dy = g.Grid.Height - dy
	}

	return dx + dy
}

// evaluateDirection determina il valore (-1, 0, 1) per una direzione
func (g *Game) evaluateDirection(dirDist, currentDist, minDist int) float64 {
	if dirDist >= currentDist {
		return -1.0 // La direzione ci allontana dal cibo
	}
	if dirDist == minDist {
		return 1.0 // È una delle direzioni ottimali
	}
	return 0.0 // È una direzione valida ma non ottimale
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
