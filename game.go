package main

import (
	"sync"
	"time"

	"golang.org/x/exp/rand"
)

type Point struct {
	X, Y int
}

type Grid struct {
	Width  int
	Height int
}

type Color struct {
	R, G, B uint8
}

type Snake struct {
	Body              []Point
	Direction         Point
	Score             int
	Dead              bool
	GameOver          bool
	Mutex             sync.RWMutex
	Color             Color
	previousHead      Point
	LastCollisionType CollisionType
}

// CollisionType represents the type of collision
type CollisionType int

const (
	NoCollision CollisionType = iota
	WallCollision
	SelfCollision
)

type Game struct {
	Grid       Grid
	snake      *Snake
	food       Point
	Steps      int
	StartTime  time.Time
	Stats      *GameStats
	lastAction int // 0: sinistra, 1: avanti, 2: destra
}

func NewSnake(startPos Point, color Color) *Snake {
	return &Snake{
		Body:              []Point{startPos},
		Direction:         Direction(rand.Intn(4) + 1).ToPoint(), // Start moving right using Direction type
		Score:             0,
		Dead:              false,
		GameOver:          false,
		Color:             color,
		previousHead:      startPos,
		LastCollisionType: NoCollision,
	}
}

// getPlayableArea returns the current playable area boundaries based on snake length
func (g *Game) getPlayableArea() (minX, maxX, minY, maxY int) {
	snakeLength := 1
	if g.snake != nil {
		snakeLength = len(g.snake.Body)
	}

	areaSize := 4 + (2 * (snakeLength - 1)) // Starts with 9x9 area

	centerX := g.Grid.Width / 2
	centerY := g.Grid.Height / 2

	minX = maxInt(0, centerX-areaSize/2)
	maxX = minInt(g.Grid.Width-1, centerX+areaSize/2)
	minY = maxInt(0, centerY-areaSize/2)
	maxY = minInt(g.Grid.Height-1, centerY+areaSize/2)

	return
}

func NewGame(width, height int) *Game {
	game := &Game{
		Grid: Grid{
			Width:  width,
			Height: height,
		},
		snake:      &Snake{},
		food:       Point{X: width / 2, Y: height / 2}, // Food starts at center
		Steps:      0,
		StartTime:  time.Now(),
		Stats:      NewGameStats(),
		lastAction: 1, // Inizia andando avanti
	}

	// Get initial playable area
	minX, maxX, minY, maxY := game.getPlayableArea()

	// Create snake with random position within playable area
	startPos := Point{
		X: minX + rand.Intn(maxX-minX+1),
		Y: minY + rand.Intn(maxY-minY+1),
	}

	// Ensure snake doesn't spawn on food
	for startPos == game.food {
		startPos = Point{
			X: minX + rand.Intn(maxX-minX+1),
			Y: minY + rand.Intn(maxY-minY+1),
		}
	}

	color := Color{
		R: uint8(rand.Intn(200) + 55),
		G: uint8(rand.Intn(200) + 55),
		B: uint8(rand.Intn(200) + 55),
	}

	game.snake = NewSnake(startPos, color)

	return game
}

// ElapsedTime restituisce la durata corrente della partita in secondi.
func (g *Game) ElapsedTime() float64 {
	return time.Since(g.StartTime).Seconds()
}

func (s *Snake) Move(newHead Point) {
	s.previousHead = s.GetHead()
	s.Body = append(s.Body, newHead)
}

func (s *Snake) RemoveTail() {
	if len(s.Body) > 0 {
		s.Body = s.Body[1:]
	}
}

func (s *Snake) GetHead() Point {
	return s.Body[len(s.Body)-1]
}

func (s *Snake) GetPreviousHead() Point {
	return s.previousHead
}

func (s *Snake) SetDirection(dir Point) {
	// Prevent 180-degree turns
	if (dir.X != 0 && dir.X == -s.Direction.X) ||
		(dir.Y != 0 && dir.Y == -s.Direction.Y) {
		return
	}
	s.Direction = dir
}

func (g *Game) GetSnake() *Snake {
	return g.snake
}

func (g *Game) GetFood() Point {
	return g.food
}

func (g *Game) validatePositions() {
	// Verifica e correggi la posizione del serpente
	g.snake.Mutex.Lock()
	for i := 0; i < len(g.snake.Body); i++ {
		if g.snake.Body[i].X >= g.Grid.Width {
			g.snake.Body[i].X = g.Grid.Width - 1
		} else if g.snake.Body[i].X < 0 {
			g.snake.Body[i].X = 0
		}
		if g.snake.Body[i].Y >= g.Grid.Height {
			g.snake.Body[i].Y = g.Grid.Height - 1
		} else if g.snake.Body[i].Y < 0 {
			g.snake.Body[i].Y = 0
		}
	}
	g.snake.Mutex.Unlock()

	// Verifica e correggi la posizione del cibo
	if g.food.X >= g.Grid.Width {
		g.food.X = g.Grid.Width - 1
	} else if g.food.X < 0 {
		g.food.X = 0
	}
	if g.food.Y >= g.Grid.Height {
		g.food.Y = g.Grid.Height - 1
	} else if g.food.Y < 0 {
		g.food.Y = 0
	}
}

// GetLastAction restituisce l'ultima azione eseguita
func (g *Game) GetLastAction() int {
	return g.lastAction
}

// SetLastAction imposta l'ultima azione eseguita
func (g *Game) SetLastAction(action int) {
	g.lastAction = action
}

func (g *Game) Update() {
	if g.snake.Dead || g.snake.GameOver {
		return
	}

	g.Steps++             // Incrementiamo il contatore ad ogni passo
	g.validatePositions() // Verifica e correggi le posizioni

	g.snake.Mutex.Lock()
	defer g.snake.Mutex.Unlock()

	// Calculate new head position
	newHead := g.calculateNewPosition()

	// Check for collisions
	collisionType := g.checkCollision(newHead)
	if collisionType != NoCollision {
		g.snake.Dead = true
		g.snake.GameOver = true
		g.snake.LastCollisionType = collisionType
		return
	}

	// Move snake
	g.snake.Move(newHead)

	// Check for food
	if newHead == g.food {
		g.snake.Score++
		g.food = g.generateFood()
	} else {
		g.snake.RemoveTail()
	}
}

func (g *Game) calculateNewPosition() Point {
	head := g.snake.GetHead()
	newX := head.X + g.snake.Direction.X
	newY := head.Y + g.snake.Direction.Y

	return Point{X: newX, Y: newY}
}

// CollisionInfo contiene informazioni sulla collisione e la distanza
type CollisionInfo struct {
	Type     CollisionType
	Distance int // Distanza Manhattan dall'ostacolo più vicino
}

func (g *Game) checkCollision(pos Point) CollisionType {
	_, collision := g.getCollisionInfo(pos)
	return collision.Type
}

// getCollisionInfo restituisce informazioni dettagliate sulla collisione più vicina
func (g *Game) getCollisionInfo(pos Point) (Point, CollisionInfo) {
	minX, maxX, minY, maxY := g.getPlayableArea()
	info := CollisionInfo{Type: NoCollision, Distance: g.Grid.Width + g.Grid.Height} // Distanza massima possibile
	var collisionPoint Point

	// Check wall collision
	if pos.X < minX || pos.X > maxX || pos.Y < minY || pos.Y > maxY {
		info.Type = WallCollision
		// Calcola la distanza dal muro più vicino
		distX := minInt(abs(pos.X-minX), abs(pos.X-maxX))
		distY := minInt(abs(pos.Y-minY), abs(pos.Y-maxY))
		info.Distance = minInt(distX, distY)
		if pos.X < minX {
			collisionPoint = Point{X: minX, Y: pos.Y}
		} else if pos.X > maxX {
			collisionPoint = Point{X: maxX, Y: pos.Y}
		} else if pos.Y < minY {
			collisionPoint = Point{X: pos.X, Y: minY}
		} else {
			collisionPoint = Point{X: pos.X, Y: maxY}
		}
		return collisionPoint, info
	}

	// Check self collision
	for _, part := range g.snake.Body[:len(g.snake.Body)-1] {
		dist := g.getManhattanDistance(pos, part)
		if dist < info.Distance {
			info.Distance = dist
			collisionPoint = part
			if dist == 0 {
				info.Type = SelfCollision
				return collisionPoint, info
			}
		}
	}

	if info.Distance < g.Grid.Width+g.Grid.Height {
		if info.Distance == 0 {
			info.Type = SelfCollision
		}
	}

	return collisionPoint, info
}

// isAdjacent controlla se un punto è adiacente alla testa del serpente
func (g *Game) isAdjacent(pos Point) bool {
	head := g.snake.GetHead()
	dx := abs(pos.X - head.X)
	dy := abs(pos.Y - head.Y)

	// Gestisce il wrapping della griglia
	if dx > g.Grid.Width/2 {
		dx = g.Grid.Width - dx
	}
	if dy > g.Grid.Height/2 {
		dy = g.Grid.Height - dy
	}

	// È adiacente se è a distanza 1 in una delle direzioni cardinali
	return (dx == 1 && dy == 0) || (dx == 0 && dy == 1)
}

// minInt returns the smaller of two integers
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// maxInt returns the larger of two integers
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (g *Game) generateFood() Point {
	// If snake length is 1, food spawns only at center
	if len(g.snake.Body) == 1 {
		return Point{X: g.Grid.Width / 2, Y: g.Grid.Height / 2}
	}

	// For length > 1, calculate area size based on length
	// length 2 -> 3x3 area (radius 1 from center)
	// length 3 -> 5x5 area (radius 2 from center)
	// length 4 -> 7x7 area (radius 3 from center)
	radius := len(g.snake.Body) - 1
	centerX := g.Grid.Width / 2
	centerY := g.Grid.Height / 2

	for {
		// Generate random position within the NxN area
		newFood := Point{
			X: centerX - radius + rand.Intn(2*radius+1),
			Y: centerY - radius + rand.Intn(2*radius+1),
		}

		// Ensure food doesn't spawn on snake
		occupied := false
		for _, part := range g.snake.Body {
			if part == newFood {
				occupied = true
				break
			}
		}

		if !occupied {
			return newFood
		}
	}
}
