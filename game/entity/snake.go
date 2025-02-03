package entity

import (
	"snake-game/ai"
	"snake-game/game/types"
	"sync"
)

type Color struct {
	R, G, B uint8
}

type Snake struct {
	Body       []types.Point
	Direction  types.Point
	Score      int
	AI         *ai.QLearning
	LastState  ai.State
	LastAction ai.Action
	Dead       bool
	GameOver   bool
	Mutex      sync.RWMutex
	Color      Color
}

func NewSnake(startPos types.Point, agent *ai.QLearning, color Color) *Snake {
	return &Snake{
		Body:      []types.Point{startPos},
		Direction: types.Point{X: 1, Y: 0}, // Start moving right
		AI:        agent,
		Score:     0,
		Dead:      false,
		GameOver:  false,
		Color:     color,
	}
}

func (s *Snake) Move(newHead types.Point) {
	s.Body = append(s.Body, newHead)
}

func (s *Snake) RemoveTail() {
	if len(s.Body) > 0 {
		s.Body = s.Body[1:]
	}
}

func (s *Snake) GetHead() types.Point {
	return s.Body[len(s.Body)-1]
}

func (s *Snake) SetDirection(dir types.Point) {
	// Only allow left, right, or straight movement relative to current direction
	currentDir := s.Direction

	// Calculate relative direction (left, right, or straight)
	// For straight movement, dir will equal currentDir
	// For left turn: rotate 90° counter-clockwise
	// For right turn: rotate 90° clockwise
	leftTurn := types.Point{X: -currentDir.Y, Y: currentDir.X}
	rightTurn := types.Point{X: currentDir.Y, Y: -currentDir.X}

	// Get the action from the AI
	var action ai.Action
	if dir == leftTurn {
		action = ai.Left
	} else if dir == rightTurn {
		action = ai.Right
	} else {
		action = ai.Straight
	}

	// Apply the action
	switch action {
	case ai.Left:
		s.Direction = leftTurn
	case ai.Right:
		s.Direction = rightTurn
	case ai.Straight:
		// Keep current direction
		s.Direction = currentDir
	}
}
