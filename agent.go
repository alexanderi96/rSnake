package main

import (
	"fmt"

	"snake-game/qlearning"
)

type SnakeAgent struct {
	agent    *qlearning.Agent
	game     *Game
	maxScore int
}

func NewSnakeAgent(game *Game) *SnakeAgent {
	agent := qlearning.NewAgent(0.3, 0.95, 0.1) // Higher learning rate and discount for better future planning
	return &SnakeAgent{
		agent:    agent,
		game:     game,
		maxScore: 0,
	}
}

// getState returns the current state as a string
func (sa *SnakeAgent) getState() string {
	// Get danger information
	ahead, right, left := sa.game.GetDangers()

	// Get food direction relative to current direction
	foodUp, foodRight, foodDown, foodLeft := sa.game.GetFoodDirection()

	// Calculate food direction relative to snake's orientation
	var foodDir int
	switch {
	case foodUp:
		foodDir = 0 // Food is above
	case foodRight:
		foodDir = 1 // Food is to the right
	case foodDown:
		foodDir = 2 // Food is below
	case foodLeft:
		foodDir = 3 // Food is to the left
	default:
		foodDir = 4 // Should never happen
	}

	// Convert state to string representation
	// Format: foodDir_danger-ahead_danger-right_danger-left_foodDist
	foodDist := sa.getManhattanDistance(sa.game.GetSnake().GetHead(), sa.game.food)
	state := fmt.Sprintf("%d_%v_%v_%v_%d", foodDir, ahead, right, left, foodDist)
	return state
}

// Update performs one step of the agent's decision making
func (sa *SnakeAgent) Update() {
	if sa.game.GetSnake().Dead {
		return
	}

	currentState := sa.getState()
	action := sa.agent.GetAction(currentState, 4) // 4 possible actions (UP, RIGHT, DOWN, LEFT)

	// Convert action to direction
	var newDir Point
	switch action {
	case UP:
		newDir = Point{X: 0, Y: -1}
	case RIGHT:
		newDir = Point{X: 1, Y: 0}
	case DOWN:
		newDir = Point{X: 0, Y: 1}
	case LEFT:
		newDir = Point{X: -1, Y: 0}
	}

	// Store current score to calculate reward
	oldScore := sa.game.GetSnake().Score
	oldLength := len(sa.game.GetSnake().Body)

	// Apply action
	sa.game.GetSnake().SetDirection(newDir)
	sa.game.Update()

	// Calculate reward
	reward := sa.calculateReward(oldScore, oldLength)

	// Update Q-values
	newState := sa.getState()
	sa.agent.Update(currentState, action, reward, newState, 4)

	// Update max score
	if sa.game.GetSnake().Score > sa.maxScore {
		sa.maxScore = sa.game.GetSnake().Score
	}
}

// calculateReward determines the reward for the last action
func (sa *SnakeAgent) calculateReward(oldScore, oldLength int) float64 {
	snake := sa.game.GetSnake()

	if snake.Dead {
		switch snake.LastCollisionType {
		case WallCollision:
			return -150.0 // Same high penalty for wall collision as self collision
		case SelfCollision:
			return -150.0 // High penalty for hitting self
		}
	}

	if snake.Score > oldScore {
		// Reward for eating food increases with snake length
		baseReward := 200.0                           // Doubled base reward for eating food
		lengthBonus := float64(len(snake.Body)) * 5.0 // Increased length bonus
		return baseReward + lengthBonus
	}

	// Calculate if we're getting closer to or further from food
	oldDist := sa.getManhattanDistance(sa.game.GetSnake().GetPreviousHead(), sa.game.food)
	newDist := sa.getManhattanDistance(snake.GetHead(), sa.game.food)

	if newDist < oldDist {
		// Reward for moving closer to food increases with snake length
		baseReward := 15.0
		lengthBonus := float64(len(snake.Body)) * 0.5
		return baseReward + lengthBonus
	} else {
		// Penalty for moving away from food increases with snake length
		basePenalty := -20.0
		lengthPenalty := float64(len(snake.Body)) * -1.0
		return basePenalty + lengthPenalty
	}
}

// getManhattanDistance calculates the Manhattan distance between two points
func (sa *SnakeAgent) getManhattanDistance(p1, p2 Point) int {
	dx := abs(p1.X - p2.X)
	dy := abs(p1.Y - p2.Y)

	// Consider grid wrapping
	if dx > sa.game.Grid.Width/2 {
		dx = sa.game.Grid.Width - dx
	}
	if dy > sa.game.Grid.Height/2 {
		dy = sa.game.Grid.Height - dy
	}

	return dx + dy
}

// abs returns the absolute value of x
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// Reset prepares the agent for a new game while keeping learned knowledge
func (sa *SnakeAgent) Reset() {
	// Save QTable
	err := sa.agent.SaveQTable("qtable.json")
	if err != nil {
		fmt.Printf("Error saving QTable: %v\n", err)
	}

	// Create new game with same dimensions
	width := sa.game.Grid.Width
	height := sa.game.Grid.Height
	sa.game = NewGame(width, height)

	// Increment episode counter for epsilon decay
	sa.agent.IncrementEpisode()
}

// SaveQTable saves the current QTable to file
func (sa *SnakeAgent) SaveQTable() error {
	return sa.agent.SaveQTable("qtable.json")
}
