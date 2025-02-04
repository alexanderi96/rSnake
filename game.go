package main

import (
	"sync"

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
	Body      []Point
	Direction Point
	Score     int
	Dead      bool
	GameOver  bool
	Mutex     sync.RWMutex
	Color     Color
}

type Game struct {
	Grid  Grid
	snake *Snake
	food  Point
}

func NewSnake(startPos Point, color Color) *Snake {
	return &Snake{
		Body:      []Point{startPos},
		Direction: Point{X: 1, Y: 0}, // Start moving right
		Score:     0,
		Dead:      false,
		GameOver:  false,
		Color:     color,
	}
}

func NewGame(width, height int) *Game {
	grid := Grid{
		Width:  width,
		Height: height,
	}

	// Create initial snake
	startPos := Point{X: width / 4, Y: height / 2}
	color := Color{
		R: uint8(rand.Intn(200) + 55),
		G: uint8(rand.Intn(200) + 55),
		B: uint8(rand.Intn(200) + 55),
	}
	snake := NewSnake(startPos, color)

	game := &Game{
		Grid:  grid,
		snake: snake,
	}

	// Generate initial food
	game.food = game.generateFood()

	return game
}

func (s *Snake) Move(newHead Point) {
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

func (g *Game) Update() {
	if g.snake.Dead || g.snake.GameOver {
		return
	}

	g.snake.Mutex.Lock()
	defer g.snake.Mutex.Unlock()

	// Calculate new head position
	newHead := g.calculateNewPosition()

	// Check for collisions
	if g.checkCollision(newHead) {
		g.snake.Dead = true
		g.snake.GameOver = true
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
	return Point{
		X: (head.X + g.snake.Direction.X + g.Grid.Width) % g.Grid.Width,
		Y: (head.Y + g.snake.Direction.Y + g.Grid.Height) % g.Grid.Height,
	}
}

func (g *Game) checkCollision(pos Point) bool {
	// Check self collision
	for _, part := range g.snake.Body[:len(g.snake.Body)-1] {
		if pos == part {
			return true
		}
	}
	return false
}

func (g *Game) generateFood() Point {
	for {
		food := Point{
			X: rand.Intn(g.Grid.Width),
			Y: rand.Intn(g.Grid.Height),
		}
		// Check if food position is valid (not on snake)
		valid := true
		for _, part := range g.snake.Body {
			if food == part {
				valid = false
				break
			}
		}
		if valid {
			return food
		}
	}
}
