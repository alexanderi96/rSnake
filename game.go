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
	Grid      Grid
	snake     *Snake
	food      Point
	Steps     int
	StartTime time.Time
	Stats     *GameStats
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
		snake:     &Snake{},
		food:      Point{X: width / 2, Y: height / 2}, // Food starts at center
		Steps:     0,
		StartTime: time.Now(),
		Stats:     NewGameStats(),
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

func (g *Game) checkCollision(pos Point) CollisionType {
	// Get current playable area boundaries
	minX, maxX, minY, maxY := g.getPlayableArea()

	// Check wall collision against dynamic boundaries
	if pos.X < minX || pos.X > maxX || pos.Y < minY || pos.Y > maxY {
		return WallCollision
	}

	// Check self collision
	for _, part := range g.snake.Body[:len(g.snake.Body)-1] {
		if pos == part {
			return SelfCollision
		}
	}
	return NoCollision
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
	// Food is always at center
	return Point{X: g.Grid.Width / 2, Y: g.Grid.Height / 2}
}
