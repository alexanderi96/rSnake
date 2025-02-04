package main

import (
	"fmt"
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
	Grid   Grid
	snakes []*Snake
	agents []*SnakeAgent
	food   Point
}

func NewSnake(startPos Point, color Color) *Snake {
	return &Snake{
		Body:              []Point{startPos},
		Direction:         Point{X: 1, Y: 0}, // Start moving right
		Score:             0,
		Dead:              false,
		GameOver:          false,
		Color:             color,
		previousHead:      startPos,
		LastCollisionType: NoCollision,
	}
}

func NewGame(width, height int, numSnakes int) *Game {
	grid := Grid{
		Width:  width,
		Height: height,
	}

	game := &Game{
		Grid:   grid,
		snakes: make([]*Snake, numSnakes),
		agents: make([]*SnakeAgent, numSnakes),
	}

	// Initialize snakes and agents first
	game.initializeAgents(numSnakes)

	// Generate initial food after snakes are initialized
	game.food = game.generateFood()

	return game
}

func (g *Game) initializeAgents(numSnakes int) {
	startPositions := []Point{
		{X: g.Grid.Width / 4, Y: g.Grid.Height / 2},     // Left side
		{X: 3 * g.Grid.Width / 4, Y: g.Grid.Height / 2}, // Right side
		{X: g.Grid.Width / 2, Y: g.Grid.Height / 4},     // Top
		{X: g.Grid.Width / 2, Y: 3 * g.Grid.Height / 4}, // Bottom
	}

	for i := 0; i < numSnakes; i++ {
		if g.snakes[i] == nil || g.snakes[i].Dead {
			pos := startPositions[i%len(startPositions)]

			if g.agents[i] == nil {
				// If no agent exists, create new snake and agent
				color := Color{
					R: uint8(rand.Intn(200) + 55),
					G: uint8(rand.Intn(200) + 55),
					B: uint8(rand.Intn(200) + 55),
				}
				g.snakes[i] = NewSnake(pos, color)
				g.agents[i] = NewSnakeAgent(g, i)
			} else {
				// If agent exists, just reset the snake with same color
				g.snakes[i] = NewSnake(pos, g.snakes[i].Color)
				// Increment episode counter for continued learning
				g.agents[i].agent.IncrementEpisode()
			}
		}
	}
}

// UpdateAgents updates all agents and respawns dead ones
func (g *Game) UpdateAgents() {
	for i, agent := range g.agents {
		if agent == nil {
			g.initializeAgents(len(g.snakes))
			continue
		}

		snake := g.GetSnake(i)
		if snake.Dead {
			// Save QTable before respawning
			err := agent.SaveQTable()
			if err != nil {
				fmt.Printf("Error saving QTable: %v\n", err)
			}

			// Respawn the snake and create new agent
			g.initializeAgents(len(g.snakes))
		} else {
			agent.Update()
		}
	}
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

func (g *Game) GetSnake(index int) *Snake {
	if index >= 0 && index < len(g.snakes) {
		return g.snakes[index]
	}
	return nil
}

func (g *Game) GetSnakes() []*Snake {
	return g.snakes
}

func (g *Game) GetFood() Point {
	return g.food
}

func (g *Game) Update() {
	for _, snake := range g.snakes {
		if snake.Dead || snake.GameOver {
			continue
		}

		snake.Mutex.Lock()

		// Calculate new head position
		newHead := g.calculateNewPosition(snake)

		// Check for collisions
		collisionType := g.checkCollision(newHead, snake)
		if collisionType != NoCollision {
			snake.Dead = true
			snake.GameOver = true
			snake.LastCollisionType = collisionType
			snake.Mutex.Unlock()
			continue
		}

		// Move snake
		snake.Move(newHead)

		// Check for food
		if newHead == g.food {
			snake.Score++
			g.food = g.generateFood()
		} else {
			snake.RemoveTail()
		}

		snake.Mutex.Unlock()
	}
}

func (g *Game) calculateNewPosition(snake *Snake) Point {
	head := snake.GetHead()
	newX := head.X + snake.Direction.X
	newY := head.Y + snake.Direction.Y

	return Point{X: newX, Y: newY}
}

func (g *Game) checkCollision(pos Point, currentSnake *Snake) CollisionType {
	if pos.X < 0 || pos.X >= g.Grid.Width || pos.Y < 0 || pos.Y >= g.Grid.Height {
		return WallCollision
	}

	// Check self collision
	for _, part := range currentSnake.Body[:len(currentSnake.Body)-1] {
		if pos == part {
			return SelfCollision
		}
	}

	// Check collision with other snakes
	for _, snake := range g.snakes {
		if snake == currentSnake {
			continue
		}
		for _, part := range snake.Body {
			if pos == part {
				return SelfCollision // Using SelfCollision type for snake-to-snake collisions
			}
		}
	}
	return NoCollision
}

func (g *Game) generateFood() Point {
	for {
		food := Point{
			X: rand.Intn(g.Grid.Width),
			Y: rand.Intn(g.Grid.Height),
		}
		// Check if food position is valid (not on any snake)
		valid := true
		for _, snake := range g.snakes {
			for _, part := range snake.Body {
				if food == part {
					valid = false
					break
				}
			}
			if !valid {
				break
			}
		}
		if valid {
			return food
		}
	}
}
