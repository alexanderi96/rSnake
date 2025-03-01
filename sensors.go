package main

import "math"

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

	// Calcola la direzione relativa del cibo rispetto alla testa
	foodDir := Point{
		X: g.food.X - head.X,
		Y: g.food.Y - head.Y,
	}

	// Considera il wrapping della griglia
	if foodDir.X > g.Grid.Width/2 {
		foodDir.X -= g.Grid.Width
	} else if foodDir.X < -g.Grid.Width/2 {
		foodDir.X += g.Grid.Width
	}
	if foodDir.Y > g.Grid.Height/2 {
		foodDir.Y -= g.Grid.Height
	} else if foodDir.Y < -g.Grid.Height/2 {
		foodDir.Y += g.Grid.Height
	}

	// Se la nuova posizione è il cibo
	if nextPos == g.food {
		return 1.0 // 100% positivo - cibo presente
	}

	// Calcola il valore percentuale del cibo basato sulla direzione
	var foodValue float64

	// Normalizza le direzioni per il confronto
	normalizedDir := Point{
		X: dir.X,
		Y: dir.Y,
	}
	if normalizedDir.X != 0 && normalizedDir.Y != 0 {
		// Per direzioni diagonali, normalizziamo a lunghezza 1
		normalizedDir.X /= 2
		normalizedDir.Y /= 2
	}

	// Calcola quanto la direzione corrente è allineata con la direzione del cibo
	if foodDir.X == 0 && foodDir.Y == 0 {
		foodValue = 0 // Siamo sulla stessa cella del cibo
	} else {
		// Converti le direzioni in float64 per i calcoli
		dirX, dirY := float64(normalizedDir.X), float64(normalizedDir.Y)
		foodX, foodY := float64(foodDir.X), float64(foodDir.Y)

		// Normalizza la direzione del cibo
		length := math.Abs(foodX) + math.Abs(foodY)
		if length > 0 {
			foodX /= length
			foodY /= length
		}

		// Se il cibo è in diagonale, avrà componenti di 0.5, 0.5
		// Se è in una direzione cardinale, avrà una componente di 1.0 e l'altra 0

		// Calcola il prodotto scalare tra la direzione normalizzata e la direzione del cibo
		dotProduct := dirX*foodX + dirY*foodY

		// Il prodotto scalare darà:
		// 1.0 se le direzioni sono identiche
		// 0.5 se la direzione è una delle componenti di una diagonale
		// 0.0 se le direzioni sono perpendicolari
		// -0.5 o -1.0 se le direzioni sono opposte
		foodValue = dotProduct
	}

	// Se non c'è pericolo, ritorna il valore del cibo
	if dangerValue == 0 {
		return foodValue
	}

	// Combina il valore del cibo con il pericolo
	// Se il cibo è in una direzione favorevole (foodValue > 0)
	if foodValue > 0 {
		return maxFloat64(dangerValue, foodValue)
	}
	// Se il cibo è in una direzione sfavorevole (foodValue <= 0)
	return minFloat64(dangerValue, foodValue)
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

// GetStateInfo restituisce una matrice 3x8 che rappresenta lo stato del gioco
// [muri][corpo][cibo] x [backLeft, left, frontLeft, front, frontRight, right, backRight, back]
func (g *Game) GetStateInfo() []float64 {
	currentDir := g.GetCurrentDirection()
	leftDir := currentDir.TurnLeft()
	rightDir := currentDir.TurnRight()

	// Calcola i vettori per tutte le direzioni
	directions := make([]Point, 8)

	// Ordine: backLeft, left, frontLeft, front, frontRight, right, backRight, back
	directions[0] = Point{ // backLeft
		X: -currentDir.ToPoint().X + leftDir.ToPoint().X,
		Y: -currentDir.ToPoint().Y + leftDir.ToPoint().Y,
	}
	directions[1] = leftDir.ToPoint() // left
	directions[2] = Point{            // frontLeft
		X: currentDir.ToPoint().X + leftDir.ToPoint().X,
		Y: currentDir.ToPoint().Y + leftDir.ToPoint().Y,
	}
	directions[3] = currentDir.ToPoint() // front
	directions[4] = Point{               // frontRight
		X: currentDir.ToPoint().X + rightDir.ToPoint().X,
		Y: currentDir.ToPoint().Y + rightDir.ToPoint().Y,
	}
	directions[5] = rightDir.ToPoint() // right
	directions[6] = Point{             // backRight
		X: -currentDir.ToPoint().X + rightDir.ToPoint().X,
		Y: -currentDir.ToPoint().Y + rightDir.ToPoint().Y,
	}
	directions[7] = Point{ // back
		X: -currentDir.ToPoint().X,
		Y: -currentDir.ToPoint().Y,
	}

	// Crea la matrice 3x8 (appiattita come array di 24 elementi)
	state := make([]float64, 24)
	head := g.snake.GetHead()

	for i, dir := range directions {
		nextPos := Point{
			X: head.X + dir.X,
			Y: head.Y + dir.Y,
		}

		// Ottiene informazioni sulla collisione
		_, collisionInfo := g.getCollisionInfo(nextPos)

		// Muri [0-7]
		if collisionInfo.Type == WallCollision {
			state[i] = 1.0
		}

		// Corpo [8-15]
		if collisionInfo.Type == SelfCollision {
			state[i+8] = 1.0
		}

		// Cibo [16-23] - Usa GetCombinedDirectionalInfo per avere informazione direzionale costante
		foodValue := g.GetCombinedDirectionalInfo(dir)
		if foodValue > 0 {
			state[i+16] = foodValue // Valore positivo indica direzione verso il cibo
		}
	}

	return state
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
