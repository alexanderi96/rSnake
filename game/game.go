package game

import (
	"snake-game/ai"
	"snake-game/game/entity"
	"snake-game/game/manager"
	"snake-game/game/types"
	"time"

	"golang.org/x/exp/rand"
)

type Game struct {
	Grid         types.Grid
	popManager   *manager.PopulationManager
	collisionMgr *manager.CollisionManager
	stateManager *manager.StateManager
}

// GetSnakes returns the current list of snakes
func (g *Game) GetSnakes() []*entity.Snake {
	return g.popManager.GetSnakes()
}

// GetUUID returns the game's UUID
func (g *Game) GetUUID() string {
	return g.stateManager.UUID
}

// GetStats returns the current game stats and game start time
func (g *Game) GetStats() (manager.GameStats, time.Time) {
	return g.stateManager.GetStats()
}

// GetStateManager returns the game's state manager
func (g *Game) GetStateManager() *manager.StateManager {
	return g.stateManager
}

func NewGame(width, height int, previousGameID string) *Game {
	grid := types.Grid{
		Width:  width,
		Height: height,
	}

	collisionMgr := manager.NewCollisionManager(grid)
	popManager := manager.NewPopulationManager(grid)
	stateManager := manager.NewStateManager(grid, previousGameID, collisionMgr, popManager)

	game := &Game{
		Grid:         grid,
		popManager:   popManager,
		collisionMgr: collisionMgr,
		stateManager: stateManager,
	}

	// Initialize population
	popManager.InitializePopulation()

	return game
}

func (g *Game) Update() {
	// Update population manager (handles aging and internal state)
	g.popManager.Update()

	// Update population history
	g.popManager.UpdatePopulationHistory()

	// Get current snakes
	snakes := g.popManager.GetSnakes()

	// Update each snake
	for _, snake := range snakes {
		if snake == nil || snake.Dead {
			continue
		}

		g.updateSnake(snake)
	}

	// Remove dead snakes
	g.popManager.RemoveDeadSnakes()

	// Update game state
	g.stateManager.Update(snakes)
}

func (g *Game) updateSnake(snake *entity.Snake) {
	snake.Mutex.Lock()
	defer snake.Mutex.Unlock()

	if snake.GameOver {
		return
	}

	// Update AI state and get next action
	g.updateAIState(snake)

	// Calculate new head position
	newHead := g.calculateNewPosition(snake)

	// Handle movement and collisions
	isDead, collidedSnake, isHeadToHead := g.collisionMgr.HandleMovement(snake, newHead, g.popManager.GetSnakes())
	if isDead {
		if isHeadToHead {
			// Try reproduction on head-to-head collision
			if g.popManager.HandleReproduction(snake, collidedSnake) {
				return
			}
		}
		// Handle death
		g.handleSnakeDeath(snake)
		return
	}

	// Move snake
	snake.Move(newHead)

	// Handle food collisions
	if hasFood, food := g.collisionMgr.CheckFoodCollisions(newHead, g.stateManager.GetFoodList()); hasFood {
		g.handleFoodCollision(snake, food)
	} else {
		snake.RemoveTail() // Only remove tail if no food was eaten
	}
}

func (g *Game) updateAIState(snake *entity.Snake) {
	currentState := g.getSnakeState(snake)

	if snake.LastState != (ai.State{}) {
		// Update Q-values based on previous state and action
		snake.AI.Update(snake.LastState, snake.LastAction, currentState)
	}

	// Get next action from AI
	action := snake.AI.GetAction(currentState)
	snake.LastState = currentState
	snake.LastAction = action

	// Convert AI action to direction
	newDirection := g.actionToDirection(action, snake.Direction)
	snake.SetDirection(newDirection)
}

func (g *Game) calculateNewPosition(snake *entity.Snake) types.Point {
	head := snake.GetHead()
	return types.Point{
		X: head.X + snake.Direction.X,
		Y: head.Y + snake.Direction.Y,
	}
}

func (g *Game) handleSnakeDeath(snake *entity.Snake) {
	snake.Dead = true
	snake.GameOver = true
	snake.AI.GamesPlayed++

	// Update session high score
	if snake.Score > snake.SessionHigh {
		snake.SessionHigh = snake.Score
	}

	// Update all-time high score
	if snake.Score > snake.AllTimeHigh {
		snake.AllTimeHigh = snake.Score
	}

	// Update scores history
	snake.Scores = append(snake.Scores, snake.Score)
	if len(snake.Scores) > 200 { // Keep only last 200 scores
		snake.Scores = snake.Scores[1:]
	}

	// Update average score
	total := 0
	for _, score := range snake.Scores {
		total += score
	}
	if len(snake.Scores) > 0 {
		snake.AverageScore = float64(total) / float64(len(snake.Scores))
	}
}

func (g *Game) handleFoodCollision(snake *entity.Snake, food types.Point) {
	snake.Score++
	g.stateManager.RemoveFood(food)
}

func (g *Game) getSnakeState(s *entity.Snake) ai.State {
	head := s.GetHead()

	// Find nearest food
	var nearestFood types.Point
	minDist := g.Grid.Width + g.Grid.Height // Max possible distance
	for _, food := range g.stateManager.GetFoodList() {
		dist := abs(food.X-head.X) + abs(food.Y-head.Y)
		if dist < minDist {
			minDist = dist
			nearestFood = food
		}
	}

	// If no food available, use current position (snake will explore)
	if minDist == g.Grid.Width+g.Grid.Height {
		nearestFood = head
	}

	// Calculate relative food direction
	foodDir := [2]int{
		sign(nearestFood.X - head.X),
		sign(nearestFood.Y - head.Y),
	}

	// Use distance to nearest food
	foodDist := minDist

	// Check dangers in all directions
	dangers := [4]bool{
		g.checkDanger(types.Point{X: head.X, Y: head.Y - 1}, s), // Up
		g.checkDanger(types.Point{X: head.X + 1, Y: head.Y}, s), // Right
		g.checkDanger(types.Point{X: head.X, Y: head.Y + 1}, s), // Down
		g.checkDanger(types.Point{X: head.X - 1, Y: head.Y}, s), // Left
	}

	currentDir := [2]int{s.Direction.X, s.Direction.Y}
	return ai.NewState(foodDir, foodDist, dangers, currentDir)
}

func (g *Game) checkDanger(pos types.Point, snake *entity.Snake) bool {
	hasCollision, _ := g.collisionMgr.CheckCollision(pos, g.popManager.GetSnakes(), snake)
	return hasCollision
}

func (g *Game) actionToDirection(action ai.Action, currentDir types.Point) types.Point {
	switch action {
	case ai.Forward:
		return currentDir
	case ai.ForwardRight:
		// Rotate current direction 90 degrees clockwise
		return types.Point{X: -currentDir.Y, Y: currentDir.X}
	case ai.ForwardLeft:
		// Rotate current direction 90 degrees counter-clockwise
		return types.Point{X: currentDir.Y, Y: -currentDir.X}
	}
	return currentDir
}

func sign(x int) int {
	if x > 0 {
		return 1
	} else if x < 0 {
		return -1
	}
	return 0
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func (g *Game) NewSnake(x, y int, agent *ai.QLearning) *entity.Snake {
	startPos := types.Point{X: x, Y: y}
	// Generate a random color for the snake
	color := entity.Color{
		R: uint8(rand.Intn(200) + 55), // Avoid too dark colors
		G: uint8(rand.Intn(200) + 55),
		B: uint8(rand.Intn(200) + 55),
	}
	snake := entity.NewSnake(startPos, agent, color)
	// Add snake to population first so food generation can consider its position
	g.popManager.AddSnake(snake)
	// Generate initial food position for the snake
	initialFood := g.stateManager.GenerateFood(g.popManager.GetSnakes())
	g.stateManager.AddFood(initialFood)
	return snake
}

func (g *Game) SaveGameStats() {
	g.stateManager.SaveGameStats()
}
