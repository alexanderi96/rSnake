package main

import (
	"encoding/binary"
	"math/rand"
	"os"
	"time"

	"snake-game/ai"

	rl "github.com/gen2brain/raylib-go/raylib"
)

const (
	width        = 40
	height       = 20
	cellSize     = 20
	screenWidth  = width * cellSize
	screenHeight = height * cellSize
	graphHeight  = 100
	graphWidth   = screenWidth - 20
	maxScores    = 50 // Maximum number of scores to show in graph
)

type Point struct {
	x, y int
}

type Game struct {
	snake            []Point
	food             Point
	direction        Point
	score            int
	sessionHighScore int
	allTimeHighScore int
	gameScores       []int
	avgScore         float64
	gameOver         bool
	ai               *ai.QLearning
	aiMode           bool
	lastState        ai.State
	lastAction       ai.Action
}

func newGame() *Game {
	// Initialize snake in the middle of the screen
	snake := []Point{
		{x: width / 2, y: height / 2},
	}

	// Load all-time high score
	allTimeHigh := 0
	if data, err := os.ReadFile("highscore.txt"); err == nil && len(data) >= 4 {
		allTimeHigh = int(binary.LittleEndian.Uint32(data))
	}

	return &Game{
		snake:            snake,
		direction:        Point{x: 1, y: 0}, // Start moving right
		food:             spawnFood(snake),
		ai:               ai.NewQLearning(),
		aiMode:           true,
		allTimeHighScore: allTimeHigh,
		gameScores:       make([]int, 0),
	}
}

func spawnFood(snake []Point) Point {
	for {
		food := Point{
			x: rand.Intn(width),
			y: rand.Intn(height),
		}

		// Check if food spawned on snake
		collision := false
		for _, p := range snake {
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

func (g *Game) getState() ai.State {
	head := g.snake[len(g.snake)-1]

	// Calculate relative food direction
	foodDir := [2]int{
		sign(g.food.x - head.x),
		sign(g.food.y - head.y),
	}

	// Calculate Manhattan distance to food
	foodDist := abs(g.food.x-head.x) + abs(g.food.y-head.y)

	// Check dangers in all directions
	dangers := [4]bool{
		g.isDanger(Point{x: head.x, y: head.y - 1}), // Up
		g.isDanger(Point{x: head.x + 1, y: head.y}), // Right
		g.isDanger(Point{x: head.x, y: head.y + 1}), // Down
		g.isDanger(Point{x: head.x - 1, y: head.y}), // Left
	}

	return ai.NewState(foodDir, foodDist, dangers)
}

func (g *Game) isDanger(p Point) bool {
	// Check wall collision
	if p.x < 0 || p.x >= width || p.y < 0 || p.y >= height {
		return true
	}

	// Check self collision
	for _, sp := range g.snake {
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

func (g *Game) update() {
	if g.gameOver {
		return
	}

	if g.aiMode {
		currentState := g.getState()

		// Get AI action if not first move
		var action ai.Action
		if g.lastState != (ai.State{}) {
			// Update Q-values
			g.ai.Update(g.lastState, g.lastAction, currentState)
		}

		// Get next action
		action = g.ai.GetAction(currentState)
		g.lastState = currentState
		g.lastAction = action

		// Convert action to direction
		switch action {
		case ai.Up:
			if g.direction.y != 1 {
				g.direction = Point{0, -1}
			}
		case ai.Right:
			if g.direction.x != -1 {
				g.direction = Point{1, 0}
			}
		case ai.Down:
			if g.direction.y != -1 {
				g.direction = Point{0, 1}
			}
		case ai.Left:
			if g.direction.x != 1 {
				g.direction = Point{-1, 0}
			}
		}
	}

	// Calculate new head position
	head := g.snake[len(g.snake)-1]
	newHead := Point{
		x: head.x + g.direction.x,
		y: head.y + g.direction.y,
	}

	// Check wall collision
	if newHead.x < 0 || newHead.x >= width || newHead.y < 0 || newHead.y >= height {
		g.gameOver = true
		return
	}

	// Check self collision
	for _, p := range g.snake {
		if p == newHead {
			g.gameOver = true
			return
		}
	}

	// Move snake
	g.snake = append(g.snake, newHead)

	// Check food collision
	if newHead == g.food {
		g.score++
		if g.score > g.sessionHighScore {
			g.sessionHighScore = g.score
		}
		if g.score > g.allTimeHighScore {
			g.allTimeHighScore = g.score
			// Save new high score
			data := make([]byte, 4)
			binary.LittleEndian.PutUint32(data, uint32(g.allTimeHighScore))
			os.WriteFile("highscore.txt", data, 0644)
		}
		g.food = spawnFood(g.snake)
	} else {
		// Remove tail if no food was eaten
		g.snake = g.snake[1:]
	}
}

func (g *Game) draw() {
	rl.BeginDrawing()
	rl.ClearBackground(rl.Black)

	// Draw grid
	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			rl.DrawRectangleLines(int32(x*cellSize), int32(y*cellSize), cellSize, cellSize, rl.DarkGray)
		}
	}

	// Draw snake
	for _, p := range g.snake {
		rl.DrawRectangle(int32(p.x*cellSize), int32(p.y*cellSize), cellSize-1, cellSize-1, rl.Green)
	}

	// Draw food
	rl.DrawRectangle(int32(g.food.x*cellSize), int32(g.food.y*cellSize), cellSize-1, cellSize-1, rl.Red)

	// Draw scores
	rl.DrawText("Current Score: "+string(rune('0'+g.score)), 10, screenHeight+10, 20, rl.White)
	rl.DrawText("Session High: "+string(rune('0'+g.sessionHighScore)), 200, screenHeight+10, 20, rl.White)
	rl.DrawText("All-Time High: "+string(rune('0'+g.allTimeHighScore)), 400, screenHeight+10, 20, rl.White)

	// Draw performance graph
	graphY := screenHeight + 40
	rl.DrawRectangleLines(10, int32(graphY), int32(graphWidth), int32(graphHeight), rl.Gray)

	// Draw graph points
	if len(g.gameScores) > 1 {
		maxScore := 0
		for _, score := range g.gameScores {
			if score > maxScore {
				maxScore = score
			}
		}
		if maxScore == 0 {
			maxScore = 1
		}

		// Draw points and connect them with lines
		for i := 1; i < len(g.gameScores); i++ {
			x1 := int32(10 + (graphWidth * (i - 1) / maxScores))
			y1 := int32(graphY + graphHeight - (graphHeight * g.gameScores[i-1] / maxScore))
			x2 := int32(10 + (graphWidth * i / maxScores))
			y2 := int32(graphY + graphHeight - (graphHeight * g.gameScores[i] / maxScore))
			rl.DrawLine(x1, y1, x2, y2, rl.Green)
			rl.DrawCircle(x2, y2, 2, rl.Yellow)
		}

		// Draw average score line
		avgY := int32(graphY + graphHeight - int(graphHeight*g.avgScore/float64(maxScore)))
		rl.DrawLine(10, avgY, int32(10+graphWidth), avgY, rl.Blue)
	}

	if g.gameOver {
		gameOverText := "Game Over! Press Q to quit (Auto-restart in 1s)"
		textWidth := rl.MeasureText(gameOverText, 20)
		rl.DrawText(gameOverText, (screenWidth-int32(textWidth))/2, screenHeight/2, 20, rl.Red)
	}

	rl.EndDrawing()
}

func main() {
	rand.Seed(time.Now().UnixNano())

	rl.InitWindow(screenWidth, screenHeight+40+graphHeight+20, "Snake AI")
	defer rl.CloseWindow()

	rl.SetTargetFPS(10)

	game := newGame()
	lastUpdate := time.Now()
	updateInterval := 100 * time.Millisecond

	for !rl.WindowShouldClose() {
		// Handle input
		if !game.aiMode {
			if rl.IsKeyDown(rl.KeyUp) && game.direction.y != 1 {
				game.direction = Point{0, -1}
			}
			if rl.IsKeyDown(rl.KeyDown) && game.direction.y != -1 {
				game.direction = Point{0, 1}
			}
			if rl.IsKeyDown(rl.KeyLeft) && game.direction.x != 1 {
				game.direction = Point{-1, 0}
			}
			if rl.IsKeyDown(rl.KeyRight) && game.direction.x != -1 {
				game.direction = Point{1, 0}
			}
		}

		if rl.IsKeyPressed(rl.KeyA) {
			game.aiMode = !game.aiMode
		}

		if rl.IsKeyPressed(rl.KeyQ) {
			game.ai.SaveQTable("qtable.json")
			break
		}

		// Update game state at fixed interval
		if time.Since(lastUpdate) >= updateInterval {
			game.update()
			lastUpdate = time.Now()

			if game.gameOver {
				// Update scores and stats
				game.gameScores = append(game.gameScores, game.score)
				if len(game.gameScores) > maxScores {
					game.gameScores = game.gameScores[1:]
				}

				// Calculate average score
				sum := 0
				for _, score := range game.gameScores {
					sum += score
				}
				game.avgScore = float64(sum) / float64(len(game.gameScores))

				// Save Q-table before restarting
				game.ai.SaveQTable("qtable.json")
				time.Sleep(time.Second)

				// Create new game but preserve scores
				oldScores := game.gameScores
				oldSessionHigh := game.sessionHighScore
				oldAvg := game.avgScore
				game = newGame()
				game.gameScores = oldScores
				game.sessionHighScore = oldSessionHigh
				game.avgScore = oldAvg
			}
		}

		game.draw()
	}

	// Save Q-table when closing window
	game.ai.SaveQTable("qtable.json")
}
