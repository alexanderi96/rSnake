package main

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	"snake-game/ai"

	rl "github.com/gen2brain/raylib-go/raylib"
)

const (
	width     = 40
	height    = 20
	maxScores = 50 // Maximum number of scores to show in graph
	numAgents = 4  // Number of AI agents
)

var (
	cellSize     int32
	screenWidth  int32
	screenHeight int32
	graphHeight  int32 // Will be set dynamically in updateDimensions()
	graphWidth   int32
	gameWidth    int32
	gameHeight   int32
	statsPanel   int32 // Width of stats panel
)

func updateDimensions() {
	// Get window dimensions
	screenWidth = int32(rl.GetScreenWidth())
	screenHeight = int32(rl.GetScreenHeight())

	// Calculate stats panel width as a percentage of screen width (20%)
	statsPanel = screenWidth / 5

	// Calculate game area dimensions (excluding stats panel)
	gameWidth = screenWidth - statsPanel
	gameHeight = screenHeight

	// Ensure game area and stats panel cover full window
	statsPanel = screenWidth - gameWidth

	// Calculate cell size based on available space and grid dimensions
	// Since we have 2x2 layout, divide the space by 2
	// Use floor division to ensure cells fit perfectly in the space
	cellW := (gameWidth - 2) / (2 * int32(width))   // -2 for padding
	cellH := (gameHeight - 2) / (2 * int32(height)) // -2 for padding
	cellSize = min(cellW, cellH)

	// Recalculate game dimensions to fit cells perfectly
	totalGridWidth := cellSize * int32(width) * 2
	totalGridHeight := cellSize * int32(height) * 2
	gameWidth = totalGridWidth + 2 // +2 for padding
	gameHeight = totalGridHeight + 2

	// Update graph dimensions
	graphWidth = statsPanel - 20
	graphHeight = screenHeight / 4 // Graph takes up 1/4 of screen height
}

func min(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}

type Point struct {
	x, y int
}

type Snake struct {
	body         []Point
	direction    Point
	color        rl.Color
	score        int
	ai           *ai.QLearning
	lastState    ai.State
	lastAction   ai.Action
	dead         bool
	food         Point
	gameOver     bool
	sessionHigh  int
	allTimeHigh  int
	scores       []int
	averageScore float64
	mutex        sync.RWMutex // Protect snake state during updates
}

type Game struct {
	snakes []*Snake
	aiMode bool
}

func newGame() *Game {
	// Initialize snakes in different positions
	snakes := make([]*Snake, numAgents)
	startPositions := [][2]int{
		{width / 4, height / 4},
		{3 * width / 4, height / 4},
		{width / 4, 3 * height / 4},
		{3 * width / 4, 3 * height / 4},
	}
	colors := []rl.Color{rl.Green, rl.Blue, rl.Purple, rl.Orange}

	// Create first agent normally (will be the parent)
	parentAI := ai.NewQLearning(0, nil, 0)
	snakes[0] = newSnake(startPositions[0], colors[0], parentAI)

	// Create other agents with mutations from parent's Q-table
	const mutationRate = 0.1 // 10% mutation rate
	for i := 1; i < numAgents; i++ {
		snakes[i] = newSnake(startPositions[i], colors[i], ai.NewQLearning(i, parentAI.QTable, mutationRate))
	}

	return &Game{
		snakes: snakes,
		aiMode: true,
	}
}

func newSnake(startPos [2]int, color rl.Color, ai *ai.QLearning) *Snake {
	// Load all-time high score for this snake
	allTimeHigh := 0
	filename := fmt.Sprintf("highscore_%d.txt", ai.ID)
	if data, err := os.ReadFile(filename); err == nil && len(data) >= 4 {
		allTimeHigh = int(binary.LittleEndian.Uint32(data))
	}

	snake := &Snake{
		body:        []Point{{x: startPos[0], y: startPos[1]}},
		direction:   Point{x: 1, y: 0}, // Start moving right
		color:       color,
		ai:          ai,
		score:       0,
		dead:        false,
		gameOver:    false,
		sessionHigh: 0,
		allTimeHigh: allTimeHigh,
		scores:      make([]int, 0),
	}
	snake.food = snake.spawnFood()
	return snake
}

func (s *Snake) spawnFood() Point {
	for {
		food := Point{
			x: rand.Intn(width),
			y: rand.Intn(height),
		}

		// Check if food spawned on snake
		collision := false
		for _, p := range s.body {
			if p == food {
				collision = true
				break
			}
		}

		if !collision {
			return food
		}
	}
}

func (s *Snake) getState() ai.State {
	head := s.body[len(s.body)-1]

	// Calculate relative food direction
	foodDir := [2]int{
		sign(s.food.x - head.x),
		sign(s.food.y - head.y),
	}

	// Calculate Manhattan distance to food
	foodDist := abs(s.food.x-head.x) + abs(s.food.y-head.y)

	// Check dangers in all directions
	dangers := [4]bool{
		s.isDanger(Point{x: head.x, y: head.y - 1}), // Up
		s.isDanger(Point{x: head.x + 1, y: head.y}), // Right
		s.isDanger(Point{x: head.x, y: head.y + 1}), // Down
		s.isDanger(Point{x: head.x - 1, y: head.y}), // Left
	}

	return ai.NewState(foodDir, foodDist, dangers)
}

func (s *Snake) isDanger(p Point) bool {
	// Check wall collision
	if p.x < 0 || p.x >= width || p.y < 0 || p.y >= height {
		return true
	}

	// Check collision with self
	for _, sp := range s.body {
		if p == sp {
			return true
		}
	}

	return false
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

func (s *Snake) update() {
	s.mutex.RLock()
	if s.gameOver || s.dead {
		s.mutex.RUnlock()
		return
	}
	s.mutex.RUnlock()

	currentState := s.getState()

	// Get AI action if not first move
	var action ai.Action
	if s.lastState != (ai.State{}) {
		// Update Q-values
		s.ai.Update(s.lastState, s.lastAction, currentState)
	}

	// Get next action
	action = s.ai.GetAction(currentState)
	s.lastState = currentState
	s.lastAction = action

	// Convert action to direction
	switch action {
	case ai.Up:
		if s.direction.y != 1 {
			s.direction = Point{0, -1}
		}
	case ai.Right:
		if s.direction.x != -1 {
			s.direction = Point{1, 0}
		}
	case ai.Down:
		if s.direction.y != -1 {
			s.direction = Point{0, 1}
		}
	case ai.Left:
		if s.direction.x != 1 {
			s.direction = Point{-1, 0}
		}
	}

	// Calculate new head position
	head := s.body[len(s.body)-1]
	newHead := Point{
		x: head.x + s.direction.x,
		y: head.y + s.direction.y,
	}

	s.mutex.Lock()
	// Check collisions
	if s.isDanger(newHead) {
		s.dead = true
		s.gameOver = true
		s.ai.GamesPlayed++

		// Update scores
		if len(s.scores) >= maxScores {
			s.scores = s.scores[1:]
		}
		s.scores = append(s.scores, s.score)

		// Calculate average score
		sum := 0
		for _, score := range s.scores {
			sum += score
		}
		s.averageScore = float64(sum) / float64(len(s.scores))
		s.mutex.Unlock()

		// Save Q-table
		filename := ai.GetQTableFilename(s.ai.ID)
		s.ai.SaveQTable(filename)
		return
	}

	// Move snake
	s.body = append(s.body, newHead)

	// Check food collision
	if newHead == s.food {
		s.score++
		if s.score > s.sessionHigh {
			s.sessionHigh = s.score
		}
		if s.score > s.allTimeHigh {
			s.allTimeHigh = s.score
			// Save new high score
			data := make([]byte, 4)
			binary.LittleEndian.PutUint32(data, uint32(s.allTimeHigh))
			filename := fmt.Sprintf("highscore_%d.txt", s.ai.ID)
			os.WriteFile(filename, data, 0644)
		}
		s.food = s.spawnFood()
	} else {
		// Remove tail if no food was eaten
		s.body = s.body[1:]
	}
	s.mutex.Unlock()
}

func (g *Game) update() {
	// Update snakes sequentially to prevent mutex contention
	for _, snake := range g.snakes {
		snake.update()

		// Handle dead snakes
		if snake.gameOver {
			time.Sleep(100 * time.Millisecond) // Small delay before reset

			snake.mutex.Lock()
			// Reset snake but keep scores
			oldScores := snake.scores
			oldAvgScore := snake.averageScore
			oldSessionHigh := snake.sessionHigh
			oldAllTimeHigh := snake.allTimeHigh
			oldMutex := snake.mutex // Keep the mutex

			*snake = *newSnake([2]int{snake.body[0].x, snake.body[0].y}, snake.color, snake.ai)
			snake.mutex = oldMutex // Restore the mutex

			snake.scores = oldScores
			snake.averageScore = oldAvgScore
			snake.sessionHigh = oldSessionHigh
			snake.allTimeHigh = oldAllTimeHigh
			snake.mutex.Unlock()

			// Save Q-table less frequently
			if snake.ai.GamesPlayed%50 == 0 {
				go func(s *Snake) {
					filename := ai.GetQTableFilename(s.ai.ID)
					s.ai.SaveQTable(filename)
				}(snake)
			}
		}
	}
}

func (g *Game) draw() {
	updateDimensions()
	rl.BeginDrawing()
	rl.ClearBackground(rl.Black)

	// Calculate dynamic sizes
	// Adjust font and line height based on available space
	fontSize := min(int32(screenHeight/40), statsPanel/15)   // Dynamic font size based on screen height and panel width
	lineHeight := min(int32(screenHeight/32), statsPanel/12) // Dynamic line height based on screen height and panel width

	// Calculate grid positions for 2x2 layout with full space utilization
	// Calculate total grid size to center the grids in each quadrant
	totalGridWidth := int32(width) * cellSize
	totalGridHeight := int32(height) * cellSize
	quadrantWidth := gameWidth / 2
	quadrantHeight := gameHeight / 2

	// Calculate offsets to center grids in quadrants
	offsetX := 1 + (quadrantWidth-totalGridWidth)/2   // +1 for left padding
	offsetY := 1 + (quadrantHeight-totalGridHeight)/2 // +1 for top padding

	gridPositions := [][2]int32{
		{offsetX, offsetY},                                          // Top-left
		{quadrantWidth + offsetX - 1, offsetY},                      // Top-right (-1 to account for center divider)
		{offsetX, quadrantHeight + offsetY - 1},                     // Bottom-left (-1 to account for center divider)
		{quadrantWidth + offsetX - 1, quadrantHeight + offsetY - 1}, // Bottom-right (-1 for both dividers)
	}

	// Draw grids and snakes for each agent
	for i, pos := range gridPositions {
		offsetX, offsetY := pos[0], pos[1]

		// Draw snake if not dead
		snake := g.snakes[i]
		snake.mutex.RLock()
		isDead := snake.dead
		score := snake.score
		if !isDead {
			// Copy values while holding lock to minimize lock time
			body := make([]Point, len(snake.body))
			copy(body, snake.body)
			food := snake.food
			color := snake.color
			snake.mutex.RUnlock()

			for _, p := range body {
				rl.DrawRectangle(
					offsetX+int32(p.x*int(cellSize)),
					offsetY+int32(p.y*int(cellSize)),
					cellSize, cellSize, color)
			}

			// Draw food for this grid
			rl.DrawRectangle(
				offsetX+int32(food.x*int(cellSize)),
				offsetY+int32(food.y*int(cellSize)),
				cellSize, cellSize, rl.Red)
		} else {
			snake.mutex.RUnlock()
		}

		// Draw grid - scale to quadrant size
		gridWidth := width
		gridHeight := height
		for x := 0; x < gridWidth; x++ {
			for y := 0; y < gridHeight; y++ {
				rl.DrawRectangleLines(
					offsetX+int32(x*int(cellSize)),
					offsetY+int32(y*int(cellSize)),
					cellSize, cellSize, rl.Gray)
			}
		}

		// Draw agent label and score
		label := fmt.Sprintf("Agent %c: %d", rune('A'+i), score)
		rl.DrawText(label, offsetX+5, offsetY+5, fontSize, snake.color)
	}

	// Stats panel on the right
	statsX := gameWidth
	statsY := int32(10)

	// Draw stats background to fully cover right side
	rl.DrawRectangle(statsX, 0, statsPanel+1, screenHeight, rl.DarkGray) // +1 to prevent gap

	// Helper function to format float with 2 decimal places
	formatFloat := func(f float64) string {
		return fmt.Sprintf("%.2f", f)
	}

	// Draw high scores for each agent
	rl.DrawText("High Scores:", statsX+10, statsY, fontSize, rl.White)
	statsY += lineHeight
	for i, snake := range g.snakes {
		snake.mutex.RLock()
		sessionHigh := snake.sessionHigh
		allTimeHigh := snake.allTimeHigh
		color := snake.color
		snake.mutex.RUnlock()

		rl.DrawText(fmt.Sprintf("Agent %c:", rune('A'+i)), statsX+20, statsY, fontSize, color)
		statsY += lineHeight
		rl.DrawText(fmt.Sprintf("Session: %d", sessionHigh), statsX+30, statsY, fontSize, color)
		statsY += lineHeight
		rl.DrawText(fmt.Sprintf("All-Time: %d", allTimeHigh), statsX+30, statsY, fontSize, color)
		statsY += lineHeight
	}
	statsY += lineHeight

	// Draw agent statistics
	rl.DrawText("Agent Stats:", statsX+10, statsY, fontSize, rl.White)
	statsY += lineHeight
	for i := 0; i < numAgents; i++ {
		currentSnake := g.snakes[i]
		currentSnake.mutex.RLock()
		avgScore := currentSnake.averageScore
		gamesPlayed := currentSnake.ai.GamesPlayed
		snakeColor := currentSnake.color
		currentSnake.mutex.RUnlock()

		rl.DrawText(fmt.Sprintf("Agent %c", rune('A'+i)), statsX+20, statsY, fontSize, snakeColor)
		statsY += lineHeight
		rl.DrawText(fmt.Sprintf("  Avg: %s", formatFloat(avgScore)), statsX+20, statsY, fontSize, snakeColor)
		statsY += lineHeight
		rl.DrawText(fmt.Sprintf("  Games: %d", gamesPlayed), statsX+20, statsY, fontSize, snakeColor)
		statsY += lineHeight + lineHeight/2
	}

	// Draw performance graph at the bottom of stats panel
	graphX := statsX + 10
	// Ensure graph stays within window bounds
	remainingSpace := screenHeight - statsY - lineHeight*2
	graphHeight = min(graphHeight, remainingSpace-lineHeight*2)
	graphY := screenHeight - graphHeight - lineHeight*2
	rl.DrawRectangleLines(graphX, graphY, statsPanel-20, graphHeight, rl.White)
	rl.DrawText("Performance", graphX, graphY-20, fontSize, rl.White)

	// Find global max score for scaling
	maxScore := 1
	for _, snake := range g.snakes {
		for _, score := range snake.scores {
			if score > maxScore {
				maxScore = score
			}
		}
	}

	// Draw scores for each agent
	graphWidth := statsPanel - 20
	for _, snake := range g.snakes {
		snake.mutex.RLock()
		scores := make([]int, len(snake.scores))
		copy(scores, snake.scores)
		avgScore := snake.averageScore
		color := snake.color
		snake.mutex.RUnlock()

		if len(scores) > 1 {
			// Draw points and connect them with lines
			for j := 1; j < len(scores); j++ {
				x1 := graphX + int32(float32(graphWidth)*float32(j-1)/float32(maxScores))
				y1 := graphY + graphHeight - int32(float32(graphHeight)*float32(scores[j-1])/float32(maxScore))
				x2 := graphX + int32(float32(graphWidth)*float32(j)/float32(maxScores))
				y2 := graphY + graphHeight - int32(float32(graphHeight)*float32(scores[j])/float32(maxScore))
				rl.DrawLine(x1, y1, x2, y2, color)
				rl.DrawCircle(x2, y2, 2, color)
			}

			// Draw average score line (dashed)
			avgY := graphY + graphHeight - int32(float32(graphHeight)*float32(avgScore)/float32(maxScore))
			for x := graphX; x < graphX+int32(graphWidth); x += 5 {
				rl.DrawLine(x, avgY, x+2, avgY, color)
			}
		}
	}

	// Draw game over text for each dead snake
	for i, snake := range g.snakes {
		snake.mutex.RLock()
		isGameOver := snake.gameOver
		color := snake.color
		snake.mutex.RUnlock()

		if isGameOver {
			gameOverText := fmt.Sprintf("Agent %c: Game Over! (Restarting...)", rune('A'+i))
			textWidth := rl.MeasureText(gameOverText, fontSize)
			pos := gridPositions[i]
			rl.DrawText(gameOverText,
				pos[0]+(totalGridWidth-int32(textWidth))/2,
				pos[1]+totalGridHeight/2,
				fontSize, color)
		}
	}

	rl.EndDrawing()
}

func main() {
	rand.Seed(time.Now().UnixNano())

	rl.InitWindow(1280, 800, "Snake AI - Q-Learning")
	rl.SetWindowMinSize(800, 600) // Set minimum window size
	rl.SetWindowState(rl.FlagWindowResizable)
	defer rl.CloseWindow()

	rl.SetTargetFPS(10)

	game := newGame()
	lastUpdate := time.Now()
	updateInterval := 100 * time.Millisecond

	for !rl.WindowShouldClose() {
		if rl.IsKeyPressed(rl.KeyQ) {
			// Save all Q-tables
			for _, snake := range game.snakes {
				filename := ai.GetQTableFilename(snake.ai.ID)
				snake.ai.SaveQTable(filename)
			}
			break
		}

		// Handle window resize
		if rl.IsWindowResized() {
			updateDimensions()
		}

		// Update game state at fixed interval
		if time.Since(lastUpdate) >= updateInterval {
			game.update()
			lastUpdate = time.Now()

			// Save Q-tables periodically
			for _, snake := range game.snakes {
				if snake.gameOver {
					go func(s *Snake) {
						filename := ai.GetQTableFilename(s.ai.ID)
						s.ai.SaveQTable(filename)
					}(snake)
				}
			}
		}

		game.draw()
	}

	// Save Q-tables sequentially when closing window to prevent I/O contention
	for _, snake := range game.snakes {
		filename := ai.GetQTableFilename(snake.ai.ID)
		snake.ai.SaveQTable(filename)
	}
}
