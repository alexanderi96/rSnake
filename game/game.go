package game

import (
	"snake-game/ai"
	"snake-game/game/entity"
	"snake-game/game/manager"
	"snake-game/game/types"

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

// GetStats returns the current game stats
func (g *Game) GetStats() manager.GameStats {
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

	// Get current state and AI action
	currentState := g.getSnakeState(snake)
	var action ai.Action

	if snake.LastState != (ai.State{}) {
		// Update Q-values based on previous state and action
		snake.AI.Update(snake.LastState, snake.LastAction, currentState)
	}

	// Get next action from AI
	action = snake.AI.GetAction(currentState)
	snake.LastState = currentState
	snake.LastAction = action

	// Convert AI action to direction
	newDirection := g.actionToDirection(action, snake.Direction)
	snake.SetDirection(newDirection)

	// Calculate new head position
	head := snake.GetHead()
	newHead := types.Point{
		X: head.X + snake.Direction.X,
		Y: head.Y + snake.Direction.Y,
	}

	// Check for collisions
	if hasCollision, collidedSnake := g.collisionMgr.CheckCollision(newHead, g.popManager.GetSnakes(), snake); hasCollision {
		// If it's a head-to-head collision, try reproduction
		if collidedSnake != nil && newHead == collidedSnake.GetHead() {
			if g.popManager.HandleReproduction(snake, collidedSnake) {
				return
			}
		}

		// Any collision that didn't result in reproduction is fatal
		snake.Dead = true
		snake.GameOver = true
		snake.AI.GamesPlayed++
		return
	}

	// Move snake
	snake.Move(newHead)

	// Check food collision with any available food
	for _, food := range g.stateManager.GetFoodList() {
		if g.collisionMgr.IsFoodCollision(newHead, food) {
			snake.Score++
			g.stateManager.RemoveFood(food)
			return // Don't remove tail when eating food
		}
	}
	snake.RemoveTail() // Only remove tail if no food was eaten
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
