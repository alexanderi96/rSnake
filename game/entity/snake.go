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
	Body              []types.Point
	Direction         types.Point
	Score             int
	AI                *ai.QLearning
	LastState         ai.State
	LastAction        ai.Action
	Dead              bool
	GameOver          bool
	SessionHigh       int
	AllTimeHigh       int
	Scores            []int
	AverageScore      float64
	Mutex             sync.RWMutex
	HasReproducedEver bool
	Color             Color
	Age               int
}

func NewSnake(startPos types.Point, agent *ai.QLearning, color Color) *Snake {
	return &Snake{
		Body:              []types.Point{startPos},
		Direction:         types.Point{X: 1, Y: 0}, // Start moving right
		AI:                agent,
		Score:             0,
		Dead:              false,
		GameOver:          false,
		Scores:            make([]int, 0),
		AverageScore:      0,
		HasReproducedEver: false,
		Color:             color,
		Age:               0,
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
	// Prevent 180-degree turns
	if (s.Direction.X != 0 && dir.X == -s.Direction.X) ||
		(s.Direction.Y != 0 && dir.Y == -s.Direction.Y) {
		return
	}
	s.Direction = dir
}
